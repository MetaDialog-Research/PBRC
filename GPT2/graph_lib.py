import abc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd

from catsample import sample_categorical

def get_graph(config, device):
    if config.graph.type == "uniform":
        return Uniform(config.tokens, config.graph.p_m, config.graph.type, config.graph.loss_type)
    elif config.graph.type == "absorb":
        return Absorbing(config.tokens, config.graph.p_m, config.graph.type, config.graph.loss_type)
    elif config.graph.type == "roulette":
        return Roulette(config.tokens, config.graph.p_m, config.graph.type, config.graph.loss_type)
    else:
        raise ValueError(f"Graph {config.graph.type} not valid")


def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class Graph(abc.ABC):

    @property
    def dim(self):
        pass

    @property
    def absorb(self):
        """
        Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask).
        """
        pass


    @abc.abstractmethod
    def rate(self, i):
        """
        Computes the i-th column of the rate matrix Q, where i is [B_1, ..., B_n].

        This is intended to compute the "forward" rate of p(X_t | X_0 = i).
        """
        pass


    @abc.abstractmethod
    def transp_rate(self, i):
        """
        Computes the i-th row of the rate matrix Q.

        Can be used to compute the reverse rate.
        """
        pass


    @abc.abstractmethod
    def transition(self, i, sigma):
        """
        Computes the i-th column of the transition matrix e^{sigma Q}.
        """
        pass


    def sample_transition(self, i, sigma):
        """
        Samples the transition vector.
        """
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")
    

    def reverse_rate(self, i, score):
        """
        Constructs the reverse rate. Which is score * transp_rate
        """
        normalized_rate = self.transp_rate(i) * score

        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def sample_rate(self, i, rate):
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)

    
    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """
        Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with
        e^{-{dsigma} E} score
        """
        pass
    

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """
        Sample the limiting distribution. Returns the probability vector as well.
        """
        pass


    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """
        Computes the score entropy function (with requisite constant normalization)
        """
        pass


class Uniform(Graph):
    """
    Everything goes to everything else. Normalized down by dimension to avoid blowup.
    """
    def __init__(self, dim, p_m, graph_type, loss_type):
        self._dim = dim
        self.p_m = p_m
        self.graph_type = graph_type
        self.loss_type = loss_type

    @property
    def dim(self):
        return self._dim
    
    @property
    def absorb(self):
        return False

    def rate(self, i):
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], - (self.dim - 1) / self.dim)
        return edge

    def transp_rate(self, i):
        return self.rate(i)

    def transition(self, i, sigma):
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma[..., None]).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans
    
    def transp_transition(self, i, sigma):
        return self.transition(i, sigma)

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        return i_pert

    def staggered_score(self, score, dsigma):
        dim = score.shape[-1]
        epow = (-dsigma).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow

    def sample_limit(self, *batch_dims):
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0):
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)

        # negative term
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        # no move means scaling by the uniform ratio. move means alter only one ratio away from 1
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )

        # constant factor
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim 
        )

        #positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        return pos_term - neg_term + const
    
    def re_score_entropy(self, score, sigma, x, x0):
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)

        log_score = score.log()
        # negative term
        neg_term = log_score.mean(dim=-1) - torch.gather(log_score, -1, x[..., None]).squeeze(-1) / self.dim
        # no move means scaling by the uniform ratio. move means alter only one ratio away from 1
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(log_score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )

        # constant factor
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim 
        )

        #positive term
        sexp = score
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        return pos_term - neg_term + const


class Absorbing(Graph):
    def __init__(self, dim, p_m, graph_type, loss_type):
        self._dim = dim
        self.p_m = p_m
        self.graph_type = graph_type
        self.loss_type = loss_type

    @property
    def dim(self):
        return self._dim + 1
    
    @property
    def absorb(self):
        return True

    def rate(self, i):
        # edge = - F.one_hot(i, num_classes=self.dim)
        # edge.scatter_add_(-1, i[..., None], torch.ones_like(edge[..., :1]))
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)        

    def transp_rate(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(self, i, sigma):
        pass
    
    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert
    
    def staggered_score(self, score, dsigma):
        score = score.clone() # yeah yeah whatever we should probably do this
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, sigma, x, x0):
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        #positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

        # constant term
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy
    
    def re_score_entropy(self, score, sigma, x, x0):
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        log_score = score.log()

        # negative_term
        neg_term = ratio * torch.gather(log_score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        #positive term
        pos_term = score[rel_ind][:, :-1].sum(dim=-1)

        # constant term
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy
    

class Roulette(Graph):
    """
    Jumping randomly into states until getting stuck in the special masked state
    """
    def __init__(self, dim, p_m, graph_type, loss_type):
        self._dim = dim
        self.p_m = p_m
        self.graph_type = graph_type
        self.loss_type = loss_type

    @property
    def dim(self):
        return self._dim + 1
    
    @property
    def absorb(self):
        return True

    def rate(self, i):
        pass


    def transp_rate(self, i):
        normal_row = torch.ones(self.dim)*(1-self.p_m)/(self.dim-1)
        normal_row[-1] = 0
        absorb_row = torch.ones(self.dim)*self.p_m
        absorb_row[-1] = 1
        rows = normal_row.reshape(1, 1, -1).to(i.device).repeat(*i.shape, 1)*(i!=(self.dim-1)).reshape(*i.shape, 1)+absorb_row.reshape(1, 1, -1).to(i.device).repeat(*i.shape, 1)*(i==(self.dim-1)).reshape(*i.shape, 1)
        rows = rows - F.one_hot(i, num_classes=self.dim)
        return rows


    def transition(self, i, sigma):
        pass
    
    def transp_transition(self, i, sigma):
        sigma = sigma[..., None]
        em = torch.exp(-sigma*self.p_m)
        eg = torch.exp(-sigma*(1-self.p_m))
        a = 1-em
        b = em*(1-eg)/(self.dim-1)
        c = em*(eg+(1-eg)/(self.dim-1))
        nonmasked_edges = (c-b)*F.one_hot(i, num_classes=self.dim)+b
        nonmasked_edges[:, :, self.dim-1] = 0

        edge = torch.where((i != self.dim - 1)[..., None], nonmasked_edges, a+(1-a)*F.one_hot(i, num_classes=self.dim))

        
        return edge

    def sample_transition(self, i, sigma):
        absorb_chance = 1 - (-sigma*self.p_m).exp()
        absorbed_indices = (torch.rand(*i.shape, device=i.device) < absorb_chance)*1
        nonabsorbed_indices = 1- absorbed_indices

        uni_move_chance = 1 - (-sigma*(1-self.p_m)).exp()
        indices_for_unisamp = torch.rand(*i.shape, device=i.device) < uni_move_chance

        uni_samp_inds = torch.where(indices_for_unisamp, torch.randint_like(i, self.dim-1), i)

        i_pert = uni_samp_inds*nonabsorbed_indices+absorbed_indices*(self.dim-1)


        return i_pert

    def staggered_score(self, score, delta_sigma):
        
        delta_sigma = delta_sigma.unsqueeze(-1)
        emi = torch.exp(delta_sigma*self.p_m)
        egi = torch.exp(delta_sigma*(1-self.p_m))
        ai = 1-emi
        bi = emi*(1-egi)/(self.dim-1)
        ci = emi*(egi+(1-egi)/(self.dim-1))
        score = score.clone()
        score_sum = score.sum(dim=-1).unsqueeze(-1)
        mod_score = score_sum * bi + score*(ci-bi) - bi*score[:,:,-1].unsqueeze(-1)
        mod_score[..., -1] += ((ai-bi)*score_sum+(1-ai+2*bi-ci)*score[:,:,-1].unsqueeze(-1)).squeeze(-1)
        return mod_score


    def sample_limit(self, *batch_dims):
        #return torch.randint(0, self.dim, batch_dims)
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, sigma, x, x0):
        # a = 1-torch.exp(-sigma/self.m)
        # b = torch.exp(-sigma/self.m)*(1-torch.exp(-sigma*(1-1/self.m)))/(self.dim-1)
        # c = 1-a-(self.dim-2)*b
        
        g=1-self.p_m

        sg = torch.expm1(sigma*g)
        sm = torch.expm1(sigma*self.p_m)

        r_ba=sg/(sm * torch.exp(sigma*g) * (self.dim-1))
        r_ca = torch.exp(-sigma*g)*(1+sg/(self.dim-1))/sm
        r_bc = sg/(torch.exp(sigma*g)+self.dim-2)#/torch.exp(sigma)

        r_cb = 1/r_bc

        # negative term
        score = torch.scatter(score, -1, (x*0+self.dim-1)[..., None], torch.zeros_like(score[..., :1]))

        neg_term = score.sum(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1)

        neg_term = torch.where(
            x==(self.dim-1),
            r_ba * neg_term+torch.gather(score, -1, x0[..., None]).squeeze(-1)*(r_ca-r_ba),
            neg_term
        )
        neg_term = torch.where(
            x==x0,
            r_bc * neg_term,
            neg_term
        )
        neg_term = torch.where(
            torch.logical_and(x != 50257, x != x0),
            neg_term+torch.gather(score, -1, x0[..., None]).squeeze(-1)*(r_cb-1) ,
            neg_term
        )

        # constant factor
        const = torch.where(
            x==(self.dim-1),
            ((self.dim - 2) ) * (r_ba) * ((r_ba).log() - 1)   +       (r_ca) * ((r_ca).log() - 1),
            0
        )
        const = torch.where(
            x==x0,
            ((self.dim - 2) ) * (r_bc) * ((r_bc).log() - 1),
            const
        )
        const = torch.where(
            torch.logical_and(x != 50257, x != x0),
            -((self.dim - 3) ) +(r_cb) * ((r_cb).log() - 1),
            const
        )
        #positive term
        sc_exp = score.exp()
        sc_exp = torch.scatter(sc_exp, -1, (x*0+self.dim-1)[..., None], torch.zeros_like(sc_exp[..., :1]))
        pos_term = sc_exp.sum(dim=-1) - torch.gather(sc_exp, -1, x[..., None]).squeeze(-1) 
        loss = (pos_term - neg_term + const)
        q = ((x==self.dim-1)*self.p_m)   +   ((x!=self.dim-1)*(1-self.p_m)/(self.dim-1))
        return loss*q
    

    def re_score_entropy(self, score, sigma, x, x0):
        # a = 1-torch.exp(-sigma/self.m)
        # b = torch.exp(-sigma/self.m)*(1-torch.exp(-sigma*(1-1/self.m)))/(self.dim-1)
        # c = 1-a-(self.dim-2)*b
        
        g=1-self.p_m

        sg = torch.expm1(sigma*g)
        sm = torch.expm1(sigma*self.p_m)

        r_ba=sg/(sm * torch.exp(sigma*g) * (self.dim-1))
        r_ca = torch.exp(-sigma*g)*(1+sg/(self.dim-1))/sm
        r_bc = sg/(torch.exp(sigma*g)+self.dim-2)#/torch.exp(sigma)

        r_cb = 1/r_bc

        # negative term
        log_score = score.log()
        log_score = torch.scatter(log_score, -1, (x*0+self.dim-1)[..., None], torch.zeros_like(log_score[..., :1]))

        neg_term = log_score.sum(dim=-1) - torch.gather(log_score, -1, x[..., None]).squeeze(-1)

        neg_term = torch.where(
            x==(self.dim-1),
            r_ba * neg_term+torch.gather(log_score, -1, x0[..., None]).squeeze(-1)*(r_ca-r_ba),
            neg_term
        )
        neg_term = torch.where(
            x==x0,
            r_bc * neg_term,
            neg_term
        )
        neg_term = torch.where(
            torch.logical_and(x != 50257, x != x0),
            neg_term+torch.gather(log_score, -1, x0[..., None]).squeeze(-1)*(r_cb-1) ,
            neg_term
        )

        # constant factor
        const = torch.where(
            x==(self.dim-1),
            ((self.dim - 2) ) * (r_ba) * ((r_ba).log() - 1)   +       (r_ca) * ((r_ca).log() - 1),
            0
        )
        const = torch.where(
            x==x0,
            ((self.dim - 2) ) * (r_bc) * ((r_bc).log() - 1),
            const
        )
        const = torch.where(
            torch.logical_and(x != 50257, x != x0),
            -((self.dim - 3) ) +(r_cb) * ((r_cb).log() - 1),
            const
        )
        #positive term
        sc_exp = score
        sc_exp = torch.scatter(sc_exp, -1, (x*0+self.dim-1)[..., None], torch.zeros_like(sc_exp[..., :1]))
        pos_term = sc_exp.sum(dim=-1) - torch.gather(sc_exp, -1, x[..., None]).squeeze(-1) 
        loss = (pos_term - neg_term + const)
        q = ((x==self.dim-1)*self.p_m)   +   ((x!=self.dim-1)*(1-self.p_m)/(self.dim-1))
        return loss*q