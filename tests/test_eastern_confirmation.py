import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.comp.eastern_confirmation import (DATA_BUDGET, FREE_FLOOR, RESERVATION, SELECTION_SHA,
    require_admission, require_eligible, storage_guard, confirmation_decision)
from scripts.collect_eastern_confirmation import main


class EasternConfirmationTest(unittest.TestCase):
    def candidate(self):
        return dict(spec_id='SPEC-111', eligible=True,
            guards={k:True for k in ('pooled_gain','regional_points','mediterranean','missed_groups')})

    def test_each_missing_or_false_guard_fails(self):
        require_eligible(self.candidate())
        for key in self.candidate()['guards']:
            value=self.candidate();value['guards'][key]=False
            with self.assertRaises(ValueError):require_eligible(value)
            value=self.candidate();value['guards'].pop(key)
            with self.assertRaises(ValueError):require_eligible(value)

    def test_ineligible_stops_before_catalogue_or_output_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);candidate=root/'candidate.json';value=self.candidate();value['eligible']=False
            candidate.write_text(json.dumps(value));out=root/'out'
            args=['collect', '--data-root',str(root),'--candidate',str(candidate),'--out',str(out)]
            for name in ('selection','runs','reference-root','prithvi-root'):args += ['--'+name,str(root/'absent')]
            with patch('sys.argv',args),patch('scripts.collect_eastern_confirmation.pystac_client.Client.open') as catalogue:
                with self.assertRaises(ValueError):main()
                catalogue.assert_not_called()
            self.assertFalse(out.exists())

    def test_admission_does_not_replace_missing_source_data(self):
        m=dict(spec_id='SPEC-113',ready_for_evaluation=True,qc_probe=False,selection_sha256=SELECTION_SHA,
            counts=dict(samples=50,groups=20))
        require_admission(m)
        for key in ('samples','groups'):
            bad=dict(m,counts=dict(m['counts']));bad['counts'][key]-=1
            with self.assertRaises(ValueError):require_admission(bad)

    def test_storage_reservations_and_free_floor_fail_closed(self):
        storage_guard(0,RESERVATION,FREE_FLOOR+RESERVATION)
        with self.assertRaises(ValueError):storage_guard(DATA_BUDGET,1,FREE_FLOOR+RESERVATION)
        with self.assertRaises(ValueError):storage_guard(0,RESERVATION,FREE_FLOOR+RESERVATION-1)

    def test_fresh_confirmation_needs_every_competitor_and_recall_guard(self):
        comparisons={k:dict(delta=.02,ci95=[.01,.04]) for k in ('post','c1mm','prithvi','reserve')}
        missed=dict(pair=0,c1mm=0)
        self.assertTrue(confirmation_decision(comparisons,missed))
        comparisons['prithvi']['ci95']=[-.001,.04]
        self.assertFalse(confirmation_decision(comparisons,missed))
        comparisons['prithvi']['ci95']=[.01,.04];missed['pair']=1
        self.assertFalse(confirmation_decision(comparisons,missed))


if __name__=='__main__':unittest.main()
