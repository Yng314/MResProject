import unittest

import torch

from vindr_mobilenet_oof import LABELS, build_parser, create_mobilenet


class VinDrMobileNetOOFTests(unittest.TestCase):
    def test_model_uses_one_input_channel_and_six_outputs(self):
        model = create_mobilenet()
        self.assertEqual(model.features[0][0].in_channels, 1)
        self.assertEqual(model.classifier[-1].out_features, len(LABELS))
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 1, 64, 64))
        self.assertEqual(tuple(output.shape), (1, len(LABELS)))

    def test_locked_training_defaults_match_project_mobilenet_oof(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--blind-cohort", "blind.csv",
                "--image-index", "index.csv",
                "--image-root", "images",
                "--output-dir", "output",
                "--seed", "13",
            ]
        )
        self.assertEqual(args.n_splits, 4)
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.early_stopping_patience, 10)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.learning_rate, 1e-3)
        self.assertEqual(args.num_workers, 4)


if __name__ == "__main__":
    unittest.main()
