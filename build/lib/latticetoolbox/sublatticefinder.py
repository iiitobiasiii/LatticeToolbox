import numpy as np
from warnings import warn

class TriangleSplitter():
    def __init__(self, kpoint):
        self.l = [[],[],[]]
        self.list_of_indices = []
        self.kpoint = kpoint

    def __call__(self, list_of_indices, **kwargs):
        self.list_of_indices = list_of_indices
        return self.split_triangles()

    def split_triangles(self):

        # if N%3=0 and the lattice has the kpoint, default method!
        if (len(self.list_of_indices) % 3 == 0) and self.kpoint:
            self.l_app(self.list_of_indices, 1, 0)
            l_new = self.triangle_indices_from_basepoints(self.list_of_indices)
            for sublist in l_new:
                try:
                    assert (len(sublist) == len(self.list_of_indices) / 3)
                    assert (self.is_disjoint(sublist))
                except AssertionError:

                    print('length', len(sublist))
                    print('is disjoint:', self.is_disjoint(sublist))
                    raise AssertionError
            return l_new

        # if not kpoint or N%3 != 0
        else:
            print("Take care about triangle splitter")
            self.l_app(self.list_of_indices, 1, 0)
            l_new = self.triangle_indices_from_basepoints(self.list_of_indices)
            for sublist in l_new:
                assert (self.is_disjoint(sublist))

            return l_new

    def other_two_sites_from_basepoints(self, triangle_indices, basepoint):
        """
        Given a list of lists (sublists are triples) return the second and third element when the first is given
        :param triangle_indices: e.g. [[1, 5, 4], [3, 7, 6], [9, 8, 7], [13, 17, 16], [6, 11, 10]]
        :param basepoint: e.g. 3
        :return: e.g. 7, 6
        """
        for c, a in enumerate(triangle_indices):
            if a[0] == basepoint:
                return a[1], a[2]
        else:
            raise IndexError

    def l_app(self, list_of_indices, basepoint, list_index):
        """

        """
        if basepoint not in self.l[list_index]:
            self.l[list_index].append(basepoint)
            b, c = self.other_two_sites_from_basepoints(list_of_indices, basepoint)
            self.l_app(list_of_indices, b, (list_index + 1) % 3)
            self.l_app(list_of_indices, c, (list_index + 2) % 3)

    def is_disjoint(self, list_of_lists):
        flattened = [a for sublist in list_of_lists for a in sublist]
        uniques = np.unique(flattened)
        return len(flattened) == len(uniques)

    def triangle_indices_from_basepoints(self, triangle_list):
        """
        For the list of lists of triangle basepoints construct the list of lists of all triangle indices
        :param triangle_list:
        :return:
        """
        new_l = []
        for sublist in self.l:
            new_l.append([[leading_element, *self.other_two_sites_from_basepoints(triangle_list, leading_element)] for leading_element in sublist])
        return new_l


def triangle_splitter(list_of_indices, kpoint):
    if not kpoint:
        warn('No kpoint !?!')
    x = TriangleSplitter(kpoint=kpoint)
    return x(list_of_indices)

