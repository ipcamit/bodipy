import numpy as np
from typing import List
from copy import deepcopy
import os
import subprocess

from numpy.core.numeric import ones

class GenerateBodipy:
    """
    Collection of all the data and routines needed to generate a PM7 optimized
    3D BODIPY molecule, needed for SLATM descriptor generation.
    """

    def __init__(self,
                fragment_file="data/all_shifted_groups.xyz", 
                file_out='bodipy_out.xyz',
                obabel="obabel",
                mopac="/opt/mopac/MOPAC2016.exe"):
        self.fragment_file = fragment_file
        self.file_out = file_out
        self.mopac = mopac
        self.obabel=obabel
        self.groups_list ={
            '000000': 0,  'H'           : 0,
            '000001': 1,  'CH3'         : 1,
            '000010': 2,  'NH2'         : 2,
            '000011': 3,  'OH'          : 3,
            '000100': 4,  'F'           : 4,
            '000101': 5,  'C=CH'        : 5,
            '000110': 6,  'CH=CH2'      : 6,
            '000111': 7,  'CN'          : 7,
            '001000': 8,  'CH=NH'       : 8,
            '001001': 9,  'CHO'         : 9,
            '001010':10,  'CH2CH3'      :10,
            '001011':11,  'CH2NH2'      :11,
            '001100':12,  'CH2F'        :12,
            '001101':13,  'NC'          :13,
            '001110':14,  'CH2OH'       :14,
            '001111':15,  'OCH3'        :15,
            '010000':16,  'CH2C=CH'     :16,
            '010001':17,  'C=CCH3'      :17,
            '010010':18,  'CH=CHCH3'    :18,
            '010011':19,  'CH2CH=CH2'   :19,
            '010100':20,  'CH2CN'       :20,
            '010101':21,  'CH2CHO'      :21,
            '010110':22,  'COCH3'       :22,
            '010111':23,  'NHCH=NH'     :23,
            '011000':24,  'C(NH2)=NH'   :24,
            '011001':25,  'N=CHNH2'     :25,
            '011010':26,  'NHCHO'       :26,
            '011011':27,  'CONH2'       :27,
            '011100':28,  'OCHO'        :28,
            '011101':29,  'COOH'        :29,
            '011110':30,  'CH2CH2CH3'   :30,
            '011111':31,  'CH(CH3)CH3'  :31,
            '100000':32,  'CH2CH2NH2'   :32,
            '100001':33,  'CH(NH2)CH3'  :33,
            '100010':34,  'NHCH2CH3'    :34,
            '100011':35,  'CH2CH2OH'    :35,
            '100100':36,  'CHF2'        :36,
            '100101':37,  'CH(OH)CH3'   :37,
            '100110':38,  'OCH2CH3'     :38,
            '100111':39,  'CH2NHCH3'    :39,
            '101000':40,  'N(CH3)CH3'   :40,
            '101001':41,  'NO2'         :41,
            '101010':42,  'CH2OCH3'     :42,
            '101011':43,  'CHCH2CH2'    :43,
            '101100':44,  'CHCH2NH'     :44,
            '101101':45,  'NCH2CH2'     :45,
            '101110':46,  'CHCH2O'      :46
        }

    @staticmethod
    def rotate(position:int, coordinates:np.ndarray) -> np.ndarray:
        """Rotate the group based on input position on where
        the group is to be placed
        Convention: S1:S7 clockwise, with S7 being the Meso group
        S7 = 0 degree
        S1, S6 = -+ 18 degree
        S2, S5 = -+ 90 degree
        S3, S4 = -+ 162 degree
        """

        if (position == 1):     angle = -18.0/180.0 * np.pi
        elif (position == 2):   angle = -90.0/180.0 * np.pi
        elif (position == 3):   angle = -162.0/180.0 * np.pi
        elif (position == 4):   angle = 162.0/180.0 * np.pi
        elif (position == 5):   angle = 90.0/180.0 * np.pi
        elif (position == 6):   angle = 18.0/180.0 * np.pi
        elif (position == 7):   angle = 0.0 # Meso position: keep unchanged
        else : raise ValueError("Wrong positional argument: {}".format(position))

        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[2, 2] = 1.0
        rotation_matrix[0, 0] = np.cos(angle + np.pi) # extra pi because in data all are rotated
        rotation_matrix[0, 1] = -np.sin(angle + np.pi)
        rotation_matrix[1, 0] = np.sin(angle + np.pi)
        rotation_matrix[1, 1] = np.cos(angle + np.pi)

        coordinates = coordinates.dot(rotation_matrix)
        return coordinates

    @staticmethod
    def shift(position:int, coordinates:np.ndarray) -> np.ndarray:
        """ Shift the group as per positions """

        if (position == 1):     shift = np.array([-0.29356614,  0.90350369, 0.])
        elif (position == 2):   shift = np.array([-0.95,        0.,         0.])
        elif (position == 3):   shift = np.array([-0.29356614, -0.90350369, 0.])
        elif (position == 4):   shift = np.array([ 0.29356614, -0.90350369, 0.])
        elif (position == 5):   shift = np.array([ 0.95,        0.,         0.])
        elif (position == 6):   shift = np.array([ 0.29356614,  0.90350369, 0.])
        elif (position == 7):   shift = np.array([ 0.,          0.95,       0.])
        else:
            raise ValueError("Wrong positional argument: {}".format(position))

        for i in range(len(coordinates)):
            coordinates[i,:] += shift
        return coordinates

    def load_bodipy_data(self):
        """ Will load core bodipy and all groups from file all_shifted_groups
        and return a list of lists with following elements [Num of atoms, [Symbols], 
        [np.array of coordinates]]. First entry is cor bodipy"""
        with open(self.fragment_file) as infile:
            num_mol = 0
            all_data = []
            while True:
                line = infile.readline()
                if len(line) == 0:
                    break
                _ = infile.readline()
                num_mol += 1
                num_atoms = int(line.split()[0])
                symbols = []
                coordinates = []
                for i in range(num_atoms):
                    data = infile.readline().split()
                    symbols.append(data[0])
                    coordinates.append(list(map(float,data[1:4])))
                coordinates = np.array(coordinates)
                all_data.append([num_atoms, symbols, coordinates])
        return all_data

    def gen_xyz(self, positions:List[int], groups:List, mode='default'):
        """ This function will take in two lists of integers
        first with positions to be substituted, second with groups
        to be substituted. and will generate a bodipy molecule with those
        positions being substituted with mntioned groups, all other positions
        will be padded to H.group can be a list of integers, binary string 
        or group composition as shown below.default -> integer, 'b' -> binary,
        'c' -> composition. other modes can be accesed using optional
        mode parameter
        """
        bodipy_data = self.load_bodipy_data()
        C_pos = [11, 12, 13, 5, 6, 7, 9]  # pos 1 2 3 4 5 6 7
        substitutents = []
        substitutents.append(bodipy_data[0])
        if mode != 'default':
            tmp_groups = []
            for group in groups:
                tmp_groups.append(self.groups_list[group])
            groups = tmp_groups

        provided_pos = set(positions)
        complete_pos = set([1,2,3,4,5,6,7])
        missing_pos = complete_pos - provided_pos
        for pos in missing_pos:
            positions.append(pos)
            groups.append(0)

        for grp, pos in zip(groups,positions):
            grp_data = bodipy_data[grp+1]
            grp_coord=deepcopy(grp_data[2])
            grp_coord = self.rotate(pos, grp_coord)
            grp_coord = self.shift(pos, grp_coord)
            for i in range(len(grp_coord)):
                grp_coord[i] += bodipy_data[0][2][C_pos[pos - 1]]
            substitutents.append(
                [bodipy_data[grp+1][0], bodipy_data[grp+1][1], grp_coord])
        with open(self.file_out,'w') as outfile:
            num_atoms = 0
            for group in substitutents:
                num_atoms += group[0]
            outfile.write(str(num_atoms)+ '\n')
            outfile.write(" ".join(list(map(str, positions))) + "|" \
                +" ".join(list(map(str, groups))) + '\n')
            for group in substitutents:
                for i, element in enumerate(group[1]):
                    outfile.write(element + "    " + \
                        "    ".join(list(map(str, group[2][i]))) + '\n')

    def opt_xyz(self):
        """
        Converts self.file_out to MOPAC file, calls mopac with 
        PM7 geom opt, coverts the output back to xyz for slatm
        cleans up afterwards. Renames the opt output file to 
        self.file_out only.
        Depends externally on MOPAC and obabel for conversiona and
        optimization. both binaries can be optionally provided to
        the initializing class instance as self. mopac, and self.obabel.
        """
        # check if file exist
        if not os.path.isfile(self.file_out):
            print("XYZ FILE NOT GENERATED, PLEASE CHECK INPUT")
            exit()

        # ======================================================
        # Run obabel
        obabel_result = subprocess.run(
            [self.obabel, "-ixyz",self.file_out, "-omopcrt", 
            "-xk PM7 CHARGE=0 CYCLES=9999","-Otmp.mop"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # check if obabel ran successfully
        # parameters: file exist, file size, "molecule converted" in STDOUT
        if not "molecule converted" in obabel_result.stderr.decode("utf-8"):
            # mol converesion failed
            print("No successful conversion message in obabel")
            exit()
        if not os.path.isfile("tmp.mop"):
            print("MOP FILE NOT GENERATED, PLEASE CHECK INPUT")
            exit()
        tmp_stat = os.stat("tmp.mop")
        if tmp_stat.st_size <= 0:
            print("MOP FILE SIZE <=0, PLEASE CHECK INPUT")
            exit()
        # If above checks ok then tmp.mop file present

        # ==================================================
        # proceed run to mopac
        mopac_result = subprocess.run([self.mopac, "tmp.mop"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # check job finished successfully by checking output of
        # mopac "joss finished successfully" and file tmp.arc
        if not "ended normally" in mopac_result.stderr.decode("utf-8"):
            # job run failed
            print("Mopac termination not normal")
            exit()
        if  (not os.path.isfile("tmp.out")) and \
            (not os.path.isfile("tmp.arc")):
            print("OUT or ARC FILE NOT GENERATED, PLEASE CHECK INPUT")
            exit()
        out_stat = os.stat("tmp.out")
        arc_stat = os.stat("tmp.arc")
        if (out_stat.st_size <= 0) or (arc_stat.st_size <= 0):
            print("OUT or ARC FILE SIZE <=0, PLEASE CHECK INPUT")
            exit()
        with open("tmp.out") as f:
            out_file_dat = f.read()
        if not "* JOB ENDED NORMALLY *" in out_file_dat:
            print("MOPAC job terminated abnormally")
            exit()

        # convert mopac output to xyz named as file_out using obabel
        # ==========================================================
        obabel_result = subprocess.run(
            [self.obabel, "-imopout", "tmp.out", "-oxyz", "-O"+self.file_out],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # check if obabel ran successfully
        # parameters: file exist, file size, "molecule converted" in STDERR
        if not "molecule converted" in obabel_result.stderr.decode("utf-8"):
            # mol converesion failed
            print("No successful conversion message in obabel")
            exit()
        tmp_stat = os.stat(self.file_out)
        if tmp_stat.st_size <= 0:
            print("MOP FILE SIZE <=0, PLEASE CHECK INPUT")
            exit()
        return True

    def cleanup(self):
        """
        cleanup after mopac
        """
        if os.path.isfile("tmp.mop"):
            os.remove("tmp.mop")
        if os.path.isfile("tmp.arc"):
            os.remove("tmp.arc")
        if os.path.isfile("tmp.out"):
            os.remove("tmp.out")

    def __call__(self, positions: List[int], groups: List):
        """
        call dunder for cleaner interface. 
        """
        self.gen_xyz(positions, groups)
        optimized = self.opt_xyz()
        if optimized:
            # if optimization was done completely then cleanup
            # else leave files for debugging
            self.cleanup()


if __name__ == "__main__":
    # mat = rotate(1, np.array([[0., 0., 0.], [0., .95, 0.], [0., 0., 0.]]))
    # mat = rotate(2, np.array([[0., 0., 0.], [0., .95, 0.], [0., 0., 0.]]))
    # mat = rotate(3, np.array([[0., 0., 0.], [0., .95, 0.], [0., 0., 0.]]))
    # mat = rotate(4, np.array([[0., 0., 0.], [0., .95, 0.], [0., 0., 0.]]))
    # mat = rotate(5, np.array([[0., 0., 0.], [0., .95, 0.], [0., 0., 0.]]))
    # mat = rotate(6, np.array([[0., 0., 0.], [0., .95, 0.], [0., 0., 0.]]))
    # mat = rotate(7, np.array([[0., 0., 0.], [0., .95, 0.], [0., 0., 0.]]))
    # rotate(8, np.eye(3))
    gen_bodipy = GenerateBodipy()
    gen_bodipy([1,7],[6,8])
