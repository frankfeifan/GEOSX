import pygeosx
import numpy as np
import os
import h5py


def recomputeSourceAndReceivers(solver, sources, receivers):
    updateSourceAndReceivers(solver, sources, receivers)

    solver.reinit()


def updateSourceAndReceivers( solver, sources_list=[], receivers_list=[] ):
    src_pos_geosx = solver.get_wrapper("sourceCoordinates").value()
    src_pos_geosx.set_access_level(pygeosx.pylvarray.RESIZEABLE)

    rcv_pos_geosx = solver.get_wrapper("receiverCoordinates").value()
    rcv_pos_geosx.set_access_level(pygeosx.pylvarray.RESIZEABLE)


    src_pos_geosx.resize(len(sources_list))
    if len(sources_list) == 0:
        src_pos_geosx.to_numpy()[:] = np.zeros((0,3))
    else:
        src_pos = [source.coords for source in sources_list]
        src_pos_geosx.to_numpy()[:] = src_pos[:]

    rcv_pos_geosx.resize(len(receivers_list))
    if len(receivers_list) == 0:
        rcv_pos_geosx.to_numpy()[:] = np.zeros((0,3))
    else:
        rcv_pos = [receiver.coords for receiver in receivers_list]
        rcv_pos_geosx.to_numpy()[:] = rcv_pos[:]

    solver.reinit()

def updateSourceValue( solver, value ):
    src_value = solver.get_wrapper("sourceValue").value()
    src_value.set_access_level(pygeosx.pylvarray.MODIFIABLE)
    src_value.to_numpy()[:] = value[:]


def residualLinearInterpolation(rtemp, maxTime, dt, dtSeismoTrace):
    r = np.zeros((int(maxTime/dt)+1, np.size(rtemp,1)))
    for i in range(np.size(rtemp,1)):
        r[:,i] = np.interp(np.linspace(0, maxTime, int(maxTime/dt)+1), np.linspace(0, maxTime, int(maxTime/dtSeismoTrace)+1), rtemp[:,i])

    return r


def resetWaveField(group):
    group.get_wrapper("Solvers/acousticSolver/indexSeismoTrace").value()[0] = 0
    nodeManagerPath = "domain/MeshBodies/mesh/Level0/nodeManager/"

    pressure_nm1 = group.get_wrapper(nodeManagerPath + "pressure_nm1").value()
    pressure_nm1.set_access_level(pygeosx.pylvarray.MODIFIABLE)

    pressure_n = group.get_wrapper(nodeManagerPath + "pressure_n").value()
    pressure_n.set_access_level(pygeosx.pylvarray.MODIFIABLE)

    pressure_np1 = group.get_wrapper(nodeManagerPath + "pressure_np1").value()
    pressure_np1.set_access_level(pygeosx.pylvarray.MODIFIABLE)

    pressure_geosx = group.get_wrapper("Solvers/acousticSolver/pressureNp1AtReceivers").value()
    pressure_geosx.set_access_level(pygeosx.pylvarray.MODIFIABLE)

    pressure_nm1.to_numpy()[:] = 0.0
    pressure_n.to_numpy()[:]   = 0.0
    pressure_np1.to_numpy()[:] = 0.0
    pressure_geosx.to_numpy()[:] = 0.0



def setTimeVariables(problem, maxTime, dt, dtSeismoTrace):
    problem.get_wrapper("Events/maxTime").value()[0] = maxTime
    problem.get_wrapper("Events/solverApplications/forceDt").value()[0] = dt
    problem.get_wrapper("/Solvers/acousticSolver/dtSeismoTrace").value()[0] = dtSeismoTrace



def computeFullGradient(directory_in_str, acquisition):
    directory = os.fsencode(directory_in_str)

    limited_aperture_flag = acquisition.limited_aperture
    nfiles = len(acquisition.shots)

    n=0
    while True:
        file_list = os.listdir(directory)
        if len(file_list) != 0:
            filename = os.fsdecode(file_list[0])
            h5p = h5py.File(os.path.join(directory_in_str,filename), "r")
            keys = list(h5p.keys())
            n += 1
            break
        else:
            continue

    h5F = h5py.File("fullGradient.hdf5", "w")
    h5F.create_dataset("fullGradient_np1", data = h5p[keys[0]], dtype='d', chunks=True, maxshape=(nfiles, None))
    h5F.create_dataset("fullGradient_np1 ReferencePosition", data = h5p[keys[-2]], chunks=True, maxshape=(nfiles, None, 3))
    h5F.create_dataset("fullGradient_np1 Time", data = h5p[keys[-1]])
    keysF = list(h5F.keys())

    h5p.close()
    os.remove(os.path.join(directory_in_str,filename))

    while True:
        file_list = os.listdir(directory)
        if n == nfiles:
            break
        elif len(file_list) > 0:
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename.startswith("partialGradient"):
                    h5p = h5py.File(os.path.join(directory_in_str, filename), 'r')
                    keysp = list(h5p.keys())

                    if limited_aperture_flag:
                        indp = 0
                        ind = []
                        for coordsp in h5p[keysp[-2]][0]:
                            indF = 0
                            flag = 0
                            for coordsF in h5F[keysF[-2]][0]:
                                if not any(coordsF - coordsp):
                                    h5F[keysF[0]][:, indF] += h5p[keysp[0]][:,indp]
                                    flag = 1
                                    break
                                indF += 1
                            if flag == 0:
                                ind.append(indp)
                            indp += 1

                        if len(ind) > 0:
                            h5F[keysF[0]].resize(h5F[keysF[0]].shape[1] + len(ind), axis=1)
                            h5F[keysF[-2]].resize(h5F[keysF[-2]].shape[1] + len(ind), axis=1)

                            h5F[keysF[0]][:, -len(ind):] = h5p[keysp[0]][:, ind]
                            h5F[keysF[-2]][:, -len(ind):] = h5p[keysp[-2]][:, ind]

                    else:
                        h5F[keysF[0]][:,:] += h5p[keysp[0]][:,:]

                    h5p.close()
                    os.remove(os.path.join(directory_in_str,filename))
                    n+=1
                else:
                    continue

        else:
            continue

    ind = np.lexsort((h5F[keysF[-2]][0][:,2], h5F[keysF[-2]][0][:,1], h5F[keysF[-2]][0][:,0]))
    h5F[keysF[-2]][:,:] = h5F[keysF[-2]][:,ind]
    h5F[keysF[0]][:,:] = h5F[keysF[0]][:,ind]

    h5F.close()
    os.rmdir(directory_in_str)

    return h5F



def print_pressure(pressure, ishot):
    print("\n" + "Pressure value at receivers for configuration " + str(ishot) + " : \n")
    print(pressure)
    print("\n")

def print_shot_config(shot_list, ishot):
    print("\n \n" + "Shot configuration number " + str(ishot) + " : \n")
    print(shot_list[ishot])

def print_group(group, indent=0):
    print("{}{}".format(" " * indent, group))

    indent += 4
    print("{}wrappers:".format(" " * indent))

    for wrapper in group.wrappers():
        print("{}{}".format(" " * (indent + 4), wrapper))
        print_with_indent(str(wrapper.value(False)), indent + 8)

    print("{}groups:".format(" " * indent))

    for subgroup in group.groups():
        print_group(subgroup, indent + 4)


def print_with_indent(msg, indent):
    indent_str = " " * indent
    print(indent_str + msg.replace("\n", "\n" + indent_str))


def print_flag(shot_list):
    i = 0
    for shot in shot_list:
        print("Shot " + str(i) + " status : " + shot.flag)
        i += 1
    print("\n")
