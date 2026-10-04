import unittest

from src.comp.eastern_confirmation_plan import REGISTRIES, balanced_frames, verify_histories
from src.comp.western_holdout import exclude_groups


class EasternPlanTest(unittest.TestCase):
    def histories(self):
        sources = {}
        for name, (count, spec, digest) in REGISTRIES.items():
            records = [dict(sample_id=str(i), bbox_wgs84=[120, 50, 121, 51],
                source_sha256='a'*64, hashes=dict(pre='a'*64)) for i in range(count)]
            sources[name] = (digest, dict(spec_id=spec, records=records, qc_probe=False,
                ready_for_training=True, ready_for_evaluation=True, selection_file='selection-v2.json',
                selection_sha256='89bba735a19cc2aad34666f6e5e3e92720113a82b5f5e96b98d571d6be1cd116'))
        return sources

    def test_all_four_complete_bound_registries_required(self):
        sources = self.histories()
        self.assertEqual(len(verify_histories(sources)), 4)
        for name in sources:
            bad = self.histories(); bad.pop(name)
            with self.assertRaises(ValueError):verify_histories(bad)
            bad = self.histories(); digest, payload = bad[name]; payload['records'].pop()
            with self.assertRaises(ValueError):verify_histories(bad)
            bad = self.histories(); bad[name] = ('b'*64, bad[name][1])
            with self.assertRaises(ValueError):verify_histories(bad)

    def test_group_purge_and_allocation_remain_deterministic(self):
        rows = [dict(sample_id=f'{g}-{i}', group_id=g, area_ha=a)
            for g in ('a','b','c') for i,a in enumerate((2,30,300,4,40))]
        self.assertEqual(balanced_frames(rows), balanced_frames(list(reversed(rows))))
        first = balanced_frames(rows, cap=3)
        self.assertEqual(len({r['group_id'] for r in first}), 3)
        self.assertEqual(len(balanced_frames(rows)), 12)
        kept, purged = exclude_groups(rows, {'a-0'})
        self.assertEqual(purged, {'a'})
        self.assertTrue(all(r['group_id'] != 'a' for r in balanced_frames(kept)))


if __name__ == '__main__':
    unittest.main()
