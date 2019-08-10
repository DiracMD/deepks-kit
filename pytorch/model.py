import torch
import torch.nn as nn 
import inspect


def log_args(name):
    def decorator(func):
        def warpper(self, *args, **kwargs):
            args_dict = inspect.getcallargs(func, self, *args, **kwargs)
            del args_dict['self']
            setattr(self, name, args_dict)
            func(self, *args, **kwargs)
        return warpper
    return decorator


class DenseNet(nn.Module):
    r"""module to create a dense network with given size, activation function and optional resnet structure

    Args:
        sizes: the shape of each layers, including the input size at begining
        actv_fn: activation function used after each layer's linear transformation
            Default: `torch.tanh`
        use_resnet: whether to use resnet structure between layers with same size or doubled size
            Default: `True`
        with_dt: whether to multiply a timestep in resnet sturcture, only effective when `use_resnet=True`
            Default: `False`
    """
    def __init__(self, sizes, actv_fn=torch.tanh, use_resnet=True, with_dt=False):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.actv_fn = actv_fn
        self.use_resnet = use_resnet
        if with_dt:
            self.dts = nn.ParameterList([nn.Parameter(torch.normal(torch.ones(out_f), std=0.1)) for out_f in sizes[1:]])
        else:
            self.dts = None

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            tmp = self.actv_fn(layer(x))
            if self.use_resnet and (tmp.shape == x.shape or tmp.shape[-1] == 2*x.shape[-1]):
                if self.dts is not None:
                    tmp = tmp * self.dts[i]
                if tmp.shape[-1] == 2*x.shape[-1]:
                    x = torch.cat([x, x], dim=-1)
                x = x + tmp
            else:
                x = tmp
        return x
    

class Descriptor(nn.Module):
    r"""module to calculate descriptor from given (projected) molecular orbitals and (baseline) energy
    
    The descriptor is given by 
    $ d_{I,i,l} = (1/N_orbit_j) * (1/N_proj) * \sum_{j, a} <i|f|R_I,a><R_I,a|f|j> g_l(e_j) $.

    Args:
        n_neuron: the shape of layers used in the filter network, input size not included
    
    Shape:
        input:
            mo_i: n_frame x n_orbit_i x n_atom x n_proj
            e_i:  n_frame x n_orbit_i
            mo_j: n_frame x n_orbit_j x n_atom x n_proj
            e_j:  n_frame x n_orbit_j
        output:
            d: n_frame x n_atom x n_orbit_i x n_filter
    """
    def __init__(self, n_neuron):
        super().__init__()
        self.filter = DenseNet([1] + n_neuron)
    
    def forward(self, mo_i, e_i, mo_j, e_j):
        nf, no_i, na, np = mo_i.shape
        nf, no_j, na, np = mo_j.shape
        g_j = self.filter(e_j.unsqueeze(-1)) # n_frame x n_orbit_j x n_filter
        u_j = torch.einsum("nof,noap->napf", g_j, mo_j) / no_j # n_frame x n_atom x n_proj x n_filter
        d = mo_i.transpose(1,2) @ u_j / np # n_frame x n_atom x n_orbit_i x n_filter
        return d


class ShellDescriptor(nn.Module):
    r"""module to calculate descriptor for each shell from (projected) MO and (baseline) energy
    
    For each shell the descriptor is given by 
    $ d_I = (1 / N_orbit) * (1 / N_shell) * \sum_{j, a} <i|f|R_I,a><R_I,a|f|j> g_l(e_j) $. 

    Args:
        n_neuron: the shape of layers used in the filter network, input size not included
        shell_sections: a list of the number of orbitals for each shell to be summed up
    
    Shape:
        input:
            mo_i: n_frame x n_orbit_i x n_atom x n_proj
            e_i:  n_frame x n_orbit_i
            mo_j: n_frame x n_orbit_j x n_atom x n_proj
            e_j:  n_frame x n_orbit_j
        output:
            d: n_frame x n_atom x n_orbit_i x (n_filter x n_shell)
    """
    def __init__(self, n_neuron, shell_sections):
        super().__init__()
        self.filter = DenseNet([1] + n_neuron)
        self.sections = shell_sections
    
    def forward(self, mo_i, e_i, mo_j, e_j):
        nf, no_i, na, np = mo_i.shape
        nf, no_j, na, np = mo_j.shape
        assert sum(self.sections) == np
        g_j = self.filter(e_j.unsqueeze(-1)) # n_frame x n_orbit_j x n_filter
        u_j = torch.einsum("nof,noap->napf", g_j, mo_j) / no_j # n_frame x n_atom x n_proj x n_filter
        u_j_list = torch.split(u_j, self.sections, dim=-2) # [n_frame x n_atom x n_ao_in_shell x n_filter] list
        mo_i_list = torch.split(mo_i.transpose(1,2), self.sections, dim=-1) # [n_frame x n_atom x n_orbit x n_ao_in_shell] list
        d_list = [mos @ us / ns for mos, us, ns in zip(mo_i_list, u_j_list, self.sections)] 
        d = torch.cat(d_list, dim=-1) # n_frame x n_atom x n_orbit_i x (n_filter x n_shell)
        return d


class QCNet(nn.Module):
    """our quantum chemistry model

    The model is given by $ E_corr = \sum_I f( d^occ(R_I), d^vir(R_I) ) $ and $d$ is calculated by `Descriptor` module.

    Args:
        n_neuron_filter: the shape of layers used in descriptor's filter
        n_neuron_fit: the shape of layers used in fitting network $f$
        shell_sections: if given, split descriptors into different shell and do summation separately
            Default: None
        e_stat: (e_avg, e_stat), if given, would scale the input energy accordingly
            Default: None
        use_resnet: whether to use resnet structure in fitting network
            Default: False

    Shape:
        input:
            mo_occ: n_frame x n_occ x n_atom x n_proj
            e_occ:  n_frame x n_occ
            mo_vir: n_frame x n_vir x n_atom x n_proj
            e_vir:  n_frame x n_vir
        output:
            e_corr: n_frame
    """
    @log_args('_init_args')
    def __init__(self, n_neuron_f, n_neuron_d, n_neuron_e, shell_sections=None, e_stat=None, use_resnet=False):
        super().__init__()
        self.fnet_occ = DenseNet([3] + n_neuron_f + [1], use_resnet=use_resnet, with_dt=True)
        self.fnet_vir = DenseNet([3] + n_neuron_f + [1], use_resnet=use_resnet, with_dt=True)
        if shell_sections is None:
            self.dnet_occ = Descriptor(n_neuron_d)
            self.dnet_vir = Descriptor(n_neuron_d)
            self.enet = DenseNet([2 * n_neuron_d[-1]] + n_neuron_e, 
                                    use_resnet=use_resnet, with_dt=True)
        else:
            self.dnet_occ = ShellDescriptor(n_neuron_d, shell_sections)
            self.dnet_vir = ShellDescriptor(n_neuron_d, shell_sections)
            self.enet = DenseNet([2 * n_neuron_d[-1] * len(shell_sections)] + n_neuron_e, 
                                    use_resnet=use_resnet, with_dt=True)
        self.final_layer = nn.Linear(n_neuron_e[-1], 1, bias=False)
        if e_stat is not None:
            self.scale_e = True
            e_avg, e_std = e_stat
            self.e_avg = nn.Parameter(torch.tensor(e_avg))
            self.e_std = nn.Parameter(torch.tensor(e_std))
        else:
            self.scale_e = False
    
    def forward(self, mo_occ, mo_vir, e_occ, e_vir):
        if self.scale_e:
            e_occ = (e_occ - self.e_avg) / self.e_std
            e_vir = (e_vir - self.e_avg) / self.e_std
        f_occ = self.fnet_occ(mo_occ).squeeze(-1)
        f_vir = self.fnet_vir(mo_vir).squeeze(-1)
        d_occ = self.dnet_occ(f_occ, e_occ, f_occ, e_occ) # n_frame x n_atom x n_occ x n_filter
        d_vir = self.dnet_vir(f_occ, e_occ, f_vir, e_vir) # n_frame x n_atom x n_occ x n_filter
        d_all = torch.cat([d_occ, d_vir], dim=-1) # n_frame x n_atom x n_occ x 2 n_filter
        e_all = self.final_layer(self.enet(d_all)) # n_frame x n_atom x n_occ x 1
        e_corr = torch.sum(e_all, dim=[1,2,3])
        return e_corr

    def save(self, filename):
        dump_dict = {
            "state_dict": self.state_dict(),
            "init_args": self._init_args
        }
        torch.save(dump_dict, filename)
    
    @staticmethod
    def load(filename):
        checkpoint = torch.load(filename, map_location="cpu")
        model = QCNet(**checkpoint["init_args"])
        model.load_state_dict(checkpoint['state_dict'])
        return model
