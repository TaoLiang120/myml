import numpy as np
from itertools import combinations
from mpi4py import MPI

from pymatgen.core.composition import Composition
from myml.myglobal import compstr2concs
from myml.models.models import NDInterpolate
from myml.myelements.myelements import thermo_scale_keys


def get_ntask_time(nproc_task, start_proc=0, thiscomm=None):
    if thiscomm is not None:
        pass
    else:
        thiscomm = MPI.COMM_WORLD
    size_local = thiscomm.Get_size()
    if size_local < nproc_task + start_proc:
        print("The number of cores must be greater than the number of communicators.")
        MPI.COMM_WORLD.Abort()
    ntask_time = int((size_local - start_proc) / nproc_task)

    return ntask_time


def split_communicator(nproc_task, start_proc=0, thiscomm=None):
    if thiscomm is not None:
        pass
    else:
        thiscomm = MPI.COMM_WORLD
    size_local = thiscomm.Get_size()
    rank_local = thiscomm.Get_rank()
    if rank_local < start_proc:
        thiscolor = size_local
    else:
        thiscolor = int((rank_local - start_proc) / nproc_task)
    thiskey = (rank_local - start_proc) % nproc_task
    comm_split = thiscomm.Split(thiscolor)
    return comm_split, thiscolor


def fill_dataframe_NDIP_parallel(compstr, key, from_ncompon, modeldata, Scale_Eform=False):
    comm_world = MPI.COMM_WORLD
    rank_world = comm_world.Get_rank()
    size_world = comm_world.Get_size()

    def compute_one_comb(itask_start, thiscolor, comm_split,
                         idtasks, thiscombs, compspace, concs, key, from_ncompon, modeldata):

        lidt = itask_start + thiscolor
        thisid = idtasks[lidt]
        comb = thiscombs[thisid]

        thiscs = []
        thisc = 0.0
        thisc4eval = []
        for i in range(from_ncompon):
            ii = comb[i]
            thiscs.append(compspace[ii])
            thisc4eval.append(concs[ii])
            thisc += concs[ii]
        thisc4eval = np.array(thisc4eval) / thisc

        thisIP = NDInterpolate(modeldata, thiscs)
        isValid = thisIP.load_model(key)
        if isValid:
            outs = thisIP.Interpolate_data(thisc4eval, key)
            thisv = outs[0] * thisc
        if isValid:
            return thisv, thisc, 1
        else:
            return thisv, thisc, 0

    comp = Composition(compstr)
    ncompon = len(comp.elements)
    if ncompon < from_ncompon:
        raise ValueError(f"{compstr} has less # of elements of NDInterpolate models.")
    rcompstr = comp.reduced_formula
    comp = Composition(rcompstr)
    compspace = []
    for el in comp.elements:
        compspace.append(el.symbol)
    concs = compstr2concs(rcompstr, reduced=True)

    inds = np.arange(ncompon, dtype=int)
    combs = combinations(inds, from_ncompon)
    thiscombs = []
    for comb in list(combs):
        thiscombs.append(comb)

    ntask_tot = len(thiscombs)
    idtasks = np.arange(ntask_tot)

    nproc_task = 1
    start_proc = 0

    ntask_time = get_ntask_time(nproc_task, start_proc=start_proc, thiscomm=None)
    comm_split, thiscolor = split_communicator(nproc_task, start_proc=start_proc, thiscomm=None)

    ntask_left = ntask_tot
    ntask_time = min(ntask_time, ntask_left)
    itask_start = 0

    thisvtot = 0.0
    thisctot = 0.0
    isValid_all = 1
    while ntask_left > 0:
        if thiscolor < ntask_time:
            thisv, thisw, isValid = compute_one_comb(itask_start, thiscolor, comm_split,
                                                     idtasks, thiscombs, compspace, concs, key, from_ncompon, modeldata)
        else:
            thisv = None
            thisw = None
            isValid = None

        if rank_world == 0:
            thisvtot += thisv
            thisctot += thisw
            isValid_all *= isValid
            for i in range(1, ntask_time):
                thisv = comm_world.recv(source=i * nproc_task, tag=i * 10 + 1)
                thisw = comm_world.recv(source=i * nproc_task, tag=i * 10 + 2)
                isValid = comm_world.recv(source=i * nproc_task, tag=i * 10 + 3)
                thisvtot += thisv
                thisctot += thisw
                isValid_all *= isValid
        else:
            if thiscolor > 0 and thiscolor < ntask_time and comm_split.Get_rank() == 0:
                comm_world.send(thisv, dest=0, tag=thiscolor * 10 + 1)
                comm_world.send(thisw, dest=0, tag=thiscolor * 10 + 2)
                comm_world.send(thisw, dest=0, tag=thiscolor * 10 + 3)
            else:
                pass

        comm_world.Barrier()

        itask_start += ntask_time
        ntask_left = ntask_left - ntask_time
        ntask_time = min(ntask_time, ntask_left)

    comm_world.Barrier()

    if rank_world == 0:
        if isValid_all == 1:
            thisvtot /= thisctot
            if Scale_Eform and key in thermo_scale_keys:
                thisvtot *= (
                            1.0 + 2.0 * (ncompon - 1) / float(ncompon) - 2.0 * (from_ncompon - 1) / float(from_ncompon))
        else:
            thisvtot = 0.0
    else:
        pass

    thisvtot = comm_world.bcast(thisvtot, root=0)
    isValid_all = comm_world.bcast(isValid_all, root=0)
    comm_split.Free()

    return thisvtot, isValid_all
