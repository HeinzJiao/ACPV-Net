import os
import os.path as osp

class DatasetCatalog(object):

    # DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__),
    #             '..','..','data'))
    # ACPV-Net project root
    PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

    DATASETS = {
        'deventer_512_train': {
            'data_dir': 'data/deventer_512'
        },
        'deventer_512_test': {
            'data_dir': 'data/deventer_512'
        },

        'deventer_512_train_with_vertex_heatmap': {
            'data_dir': 'data/deventer_512',
            'factory': 'VertexHeatmapTrainDataset'
        },
        'deventer_512_test_with_vertex_heatmap': {
            'data_dir': 'data/deventer_512',
            'factory': 'VertexHeatmapTestDataset'
        },

        'deventer_512_train_with_vertex_heatmap_afm': {
            'data_dir': 'data/deventer_512',
            'factory': 'VertexHeatmapAfmTrainDataset'
        },
        'deventer_512_test_with_vertex_heatmap_afm': {
            'data_dir': 'data/deventer_512',
            'factory': 'VertexHeatmapAfmTestDataset'
        },

        'deventer_512_train_with_latent_vertex_heatmap': {
            'data_dir': 'data/deventer_512',
            'factory': 'LatentVertexHeatmapTrainDataset'
        },
        'deventer_512_test_with_latent_vertex_heatmap': {
            'data_dir': 'data/deventer_512',
            'factory': 'LatentVertexHeatmapTestDataset'
        },

        'whu_building_train_with_latent_vertex_heatmap': {
            'data_dir': 'data/whu_building',
            'factory': 'LatentVertexHeatmapTrainDataset'
        },
        'whu_building_test_with_latent_vertex_heatmap': {
            'data_dir': 'data/whu_building',
            'factory': 'LatentVertexHeatmapTestDataset'
        },

        'shanghai_building_train_with_latent_vertex_heatmap': {
            'data_dir': 'data/shanghai',
            'factory': 'LatentVertexHeatmapTrainDataset'
        },
        'shanghai_building_test_with_latent_vertex_heatmap': {
            'data_dir': 'data/shanghai',
            'factory': 'LatentVertexHeatmapTestDataset'
        },

        'ai4smallfarms_train_with_latent_vertex_heatmap': {
            'data_dir': '../../agricultural_parcel_extraction/AI4SmallFarms_split_0.05',
            'factory': 'LatentVertexHeatmapTrainDataset'
        },
        'ai4smallfarms_test_with_latent_vertex_heatmap': {
            'data_dir': '../../agricultural_parcel_extraction/AI4SmallFarms_split_0.05',
            'factory': 'LatentVertexHeatmapTestDataset'
        },
        'ai4smallfarms_train_with_latent_vertex_heatmap_dualseg': {
            'data_dir': '../../agricultural_parcel_extraction/AI4SmallFarms_split_0.05',
            'factory': 'LatentVertexHeatmapDualSegTrainDataset'
        },
        'ai4smallfarms_test_with_latent_vertex_heatmap_dualseg': {
            'data_dir': '../../agricultural_parcel_extraction/AI4SmallFarms_split_0.05',
            'factory': 'LatentVertexHeatmapDualSegTestDataset'
        },
        'ai4smallfarms_train_with_latent_vertex_heatmap_dualseg_jointlatent': {
            'data_dir': '../../agricultural_parcel_extraction/AI4SmallFarms_split_0.05',
            'factory': 'LatentVertexHeatmapDualSegJointLatentTrainDataset'
        },
        'ai4smallfarms_test_with_latent_vertex_heatmap_dualseg_jointlatent': {
            'data_dir': '../../agricultural_parcel_extraction/AI4SmallFarms_split_0.05',
            'factory': 'LatentVertexHeatmapDualSegJointLatentTestDataset'
        },
        
        'ai4smallfarms_full_train_with_latent_vertex_heatmap': {
            'data_dir': '../AI4SmallFarms',
            'factory': 'LatentVertexHeatmapTrainDataset'
        },
        'ai4smallfarms_full_test_with_latent_vertex_heatmap': {
            'data_dir': '../AI4SmallFarms',
            'factory': 'LatentVertexHeatmapTestDataset'
        },
        'ai4smallfarms_full_train_with_latent_vertex_heatmap_dualseg': {
            'data_dir': '../AI4SmallFarms',
            'factory': 'LatentVertexHeatmapDualSegTrainDataset'
        },
        'ai4smallfarms_full_test_with_latent_vertex_heatmap_dualseg': {
            'data_dir': '../AI4SmallFarms',
            'factory': 'LatentVertexHeatmapDualSegTestDataset'
        },
        'ai4smallfarms_full_train_with_latent_vertex_heatmap_dualseg_jointlatent': {
            'data_dir': '../AI4SmallFarms',
            'factory': 'LatentVertexHeatmapDualSegJointLatentTrainDataset'
        },
        'ai4smallfarms_full_test_with_latent_vertex_heatmap_dualseg_jointlatent': {
            'data_dir': '../AI4SmallFarms',
            'factory': 'LatentVertexHeatmapDualSegJointLatentTestDataset'
        },
    }

    @staticmethod
    def get(name):
        # data_dir = DatasetCatalog.DATA_DIR

        assert name in DatasetCatalog.DATASETS
        attrs = DatasetCatalog.DATASETS[name]

        if 'train' in name:
            split = 'train'
        elif 'val' in name:
            split = 'val'
        elif 'test' in name:
            split = 'test'
        else:
            raise ValueError(f"Cannot determine split type from dataset name: {name}")
        
        args = dict(
            data_dir=osp.join(DatasetCatalog.PROJECT_ROOT, attrs['data_dir']),
            split=split
        )

        if 'factory' in attrs:
            return dict(factory=attrs['factory'], args=args)

        if 'train' in name:
            return dict(factory="TrainDataset", args=args)
        if 'test' in name:
            if 'ann_file' in attrs:
                return dict(factory="TestDatasetWithAnnotations", args=args)
            else:
                return dict(factory="TestDataset", args=args)

        raise NotImplementedError()
