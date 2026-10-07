import unittest
import cv2
import numpy as np
from component_probe import collect_components, connection_candidates

class ComponentTests(unittest.TestCase):
    def test_two_fragments_preserved_and_geometry_link_is_unverified(self):
        m=np.zeros((110,150),np.uint8)
        m[45:56,10:61]=1; m[45:56,80:131]=1
        before=m.copy(); c=collect_components(m)
        self.assertEqual(len(c),2)
        self.assertEqual([len(p["endpoints"]) for p in c],[2,2])
        links=connection_candidates(c)
        self.assertEqual(len(links),1)
        self.assertEqual(links[0]["status"],"geometry_only_occlusion_unverified")
        np.testing.assert_array_equal(m,before)

    def test_single_component_has_no_cross_component_link(self):
        m=np.zeros((100,150),np.uint8); m[45:56,10:130]=1
        c=collect_components(m)
        self.assertEqual(len(c),1)
        self.assertEqual(len(c[0]["endpoints"]),2)
        self.assertEqual(connection_candidates(c),[])

    def test_perpendicular_fragments_not_linked(self):
        m=np.zeros((110,160),np.uint8)
        m[45:56,10:60]=1; m[60:100,70:81]=1
        self.assertEqual(connection_candidates(collect_components(m)),[])

    def test_fork_keeps_all_terminals(self):
        m=np.zeros((150,150),np.uint8)
        for end in [(20,20),(130,20),(75,135)]:
            cv2.line(m,(75,75),end,1,9)
        c=collect_components(m)
        self.assertEqual(len(c[0]["endpoints"]),3)

    def test_closed_ring_does_not_invent_endpoint(self):
        m=np.zeros((150,150),np.uint8)
        cv2.circle(m,(75,75),45,1,9)
        c=collect_components(m)
        self.assertEqual(c[0]["endpoints"],[])

    def test_empty_mask_and_isolated_pixel(self):
        m=np.zeros((20,20),np.uint8)
        self.assertEqual(collect_components(m),[])
        m[10,10]=1
        self.assertEqual(collect_components(m)[0]["endpoints"],[])

if __name__=="__main__":
    unittest.main(verbosity=2)
