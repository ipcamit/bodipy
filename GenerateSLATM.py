from qml.representations import get_slatm_mbtypes
from qml import Compound
import numpy as np


class GenerateSLATM:
    """
    Simple class wrapper to initialize qml SLATM routines
    call dunder overloaded for generating slatm descriptor
    """

    def __init__(self):
        # Declare mbtypes as number of possible elements occuring in
        # BODIPY datasets
        self.mbtypes = get_slatm_mbtypes(
            [np.array([1, 1, 1, 6, 6, 6, 5, 7, 7, 7, 8, 8, 8, 9, 9, 9])])

    def __call__(self, xyz_file: str = "bodipy_out.xyz") -> np.ndarray:
        """
        Get xyz file and return slatm of size 1x18023
        """
        data = Compound(xyz=xyz_file)
        data.generate_slatm(self.mbtypes, local=False)
        return data.representation


if __name__ == "__main__":
    get_slatm = GenerateSLATM()
    x = get_slatm()
    print(x.shape)