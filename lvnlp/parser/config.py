import sys
from collections import defaultdict
from dataclasses import dataclass, field

from transformers import HfArgumentParser

ALL_TASKS = ['upos', 'xpos', 'lemma', 'feats', 'parser']


@dataclass
class Arguments:
    model: str = field(default='', metadata={'help': 'Model name or path'})
    prefix: str = field(default='', metadata={'help': 'Prefix'})
    batch_size: int = field(default=16, metadata={'help': 'Batch size'})
    lr: float = field(default=6e-4, metadata={'help': 'Learning rate'})
    weight_decay: float = field(default=1e-4, metadata={'help': 'Weight decay'})
    dropout: float = field(default=0.3, metadata={'help': 'Dropout rate'})
    label_smoothing: float = field(default=0.2, metadata={'help': 'Label smoothing'})
    epochs: int = field(default=15, metadata={'help': 'Number of epochs'})
    seed: int = field(default=42, metadata={'help': 'Random seed'})
    min_count: int = field(default=3, metadata={'help': 'Minimum count'})
    ema_decay: float = field(default=0, metadata={'help': 'EMA decay: 0, 0.995'})
    log_wandb: bool = field(default=True, metadata={'help': 'Log to Weights & Biases'})
    treebank_path: str = field(default='data/ud_2.17', metadata={'help': 'Path to UD treebanks'})
    out_dir: str = field(default='out', metadata={'help': 'Output directory'})
    wandb_project: str = field(default='ud', metadata={'help': 'W&B project'})
    lr_ratio: float = field(default=0.1, metadata={'help': 'Learning rate ratio'})
    group: str = field(default='v0', metadata={'help': 'Experiment group'})
    device: int = field(default=0, metadata={'help': 'CUDA device'})
    workers: int = field(default=7, metadata={'help': 'Number of workers'})
    random_mask: float = field(default=0.15, metadata={'help': 'Random mask'})
    precision: str = field(default='bf16', metadata={'help': 'Precision: no|fp16|bf16'})
    ffn_size: int = field(default=256, metadata={'help': 'FFN size'})
    # parser_ffn_size: int = field(default=2560, metadata={'help': 'Parser FFN size'})
    arc_mlp_size: int = field(default=512, metadata={'help': 'Hidden size of arc MLP'})
    rel_mlp_size: int = field(default=128, metadata={'help': 'Hidden size of relation MLP'})
    best_metric: str = field(default='las_xpos_lemma', metadata={'help': 'Best model selection metric mlas_blex or las_xpos_lemma'})
    tasks: str = field(default='all', metadata={'help': f'Prediction tasks: all or {ALL_TASKS}'})
    loss_weights: str = field(default='lemma=4', metadata={'help': 'Loss weight'})
    mono_xpos: bool = field(default=False, metadata={'help': 'Mono XPOS'})
    mono_feats: bool = field(default=False, metadata={'help': 'Mono feats'})
    version: str = field(default='v1', metadata={'help': 'Version string'})

    def __post_init__(self):
        if self.precision not in {'no', 'fp16', 'bf16'}:
            raise ValueError(f'Unsupported precision: {self.precision}')
        assert set(ALL_TASKS if self.tasks == 'all' else self.tasks.split(',')).issubset(set(ALL_TASKS)), f'Unsupported tasks: {self.tasks}'

    @property
    def tasks_list(self):
        return ALL_TASKS if self.tasks == 'all' else self.tasks.split(',')

    @property
    def loss_weights_dict(self):
        r = defaultdict(lambda: 1.0)
        if self.loss_weights:
            r.update((k, float(v)) for k, v in (item.split('=', 1) for item in self.loss_weights.split(',')))
        return r

    @classmethod
    def load(cls) -> 'Arguments':
        parser = HfArgumentParser(Arguments)
        if len(sys.argv) == 2 and sys.argv[1].startswith('Arguments('):
            args: Arguments = eval(sys.argv[1])
        else:
            args: Arguments = parser.parse_args_into_dataclasses()[0]
        return args
