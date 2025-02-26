import torch
import time


class FPSBenchmark():
    def __init__(
        self,
        model: torch.nn.Module,
        input_size: tuple,
        device: str = "cpu",
        warmup_num: int = 5,
        log_interval: int = 10,
        iterations: int = 100,
        repeat_num: int = 1,
    ) -> None:
        """FPS benchmark.

        Ref:
            MMDetection: https://mmdetection.readthedocs.io/en/stable/useful_tools.html#fps-benchmark.

        Args:
            model (torch.nn.Module): model to be tested.
            input_size (tuple): model acceptable input size, e.g. `BCHW`, make sure `batch_size` is 1.
            device (str): device for test. Default to "cpu".
            warmup_num (int, optional): the first several iterations may be very slow so skip them. Defaults to 5.
            iterations (int, optional): numer of iterations in a single test. Defaults to 100.
            repeat_num (int, optional): number of repeat tests. Defaults to 1.
        """
        # Parameters for `load_model`
        self.model = model
        self.input_size = input_size
        self.device = device

        # Parameters for `measure_inference_speed`
        self.warmup_num = warmup_num
        self.log_interval = log_interval
        self.iterations = iterations

        # Parameters for `repeat_measure_inference_speed`
        self.repeat_num = repeat_num

    def load_model(self):
        model = self.model.to(self.device)
        model.eval()
        return model

    def measure_inference_speed(self):
        model = self.load_model()
        pure_inf_time = 0
        fps = 0

        for i in range(self.iterations):
            input_data = torch.randn(self.input_size, device=self.device)
            if "cuda" in self.device:
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                with torch.no_grad():
                    model(input_data)
                torch.cuda.synchronize()
            elif "cpu" in self.device:
                start_time = time.perf_counter()
                with torch.no_grad():
                    model(input_data)
            else:
                NotImplementedError(
                    f"{self.device} hasn't been implemented yet."
                )
            elapsed = time.perf_counter() - start_time

            if i >= self.warmup_num:
                pure_inf_time += elapsed
                if (i + 1) % self.log_interval == 0:
                    fps = (i + 1 - self.warmup_num) / pure_inf_time
                    print(
                        f'Done image [{i + 1:0>3}/{self.iterations}], '
                        f'FPS: {fps:.2f} img/s, '
                        f'Times per image: {1000 / fps:.2f} ms/img',
                        flush=True,
                    )
                else:
                    pass
            else:
                pass
        fps = (self.iterations - self.warmup_num) / pure_inf_time
        print(
            f'Overall FPS: {fps:.2f} img/s, '
            f'Times per image: {1000 / fps:.2f} ms/img',
            flush=True,
        )
        return fps

    def repeat_measure_inference_speed(self):
        assert self.repeat_num >= 1
        fps_list = []
        for _ in range(self.repeat_num):
            fps_list.append(self.measure_inference_speed())
        if self.repeat_num > 1:
            fps_list_ = [round(fps, 2) for fps in fps_list]
            times_pre_image_list_ = [round(1000 / fps, 2) for fps in fps_list]
            mean_fps_ = sum(fps_list_) / len(fps_list_)
            mean_times_pre_image_ = sum(times_pre_image_list_) / len(
                times_pre_image_list_)
            print(
                f'Overall FPS: {fps_list_}[{mean_fps_:.2f}] img/s, '
                f'Times per image: '
                f'{times_pre_image_list_}[{mean_times_pre_image_:.2f}] ms/img',
                flush=True,
            )
            return fps_list
        else:
            return fps_list[0]

from mmyolo.registry import MODELS
from mmengine.runner import Runner, load_checkpoint
import argparse
from mmengine import Config, DictAction

def parse_args():
    parser = argparse.ArgumentParser(description='MMYOLO benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--repeat-num',
        type=int,
        default=1,
        help='number of repeat times of measurement for averaging the results')
    parser.add_argument(
        '--max-iter', type=int, default=2000, help='num of max iter')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing '
        'benchmark metrics')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg['custom_imports'])

    model = MODELS.build(cfg.model)
    # load_checkpoint(model, cfg.checkpoint, map_location='cpu')
    FPSBenchmark(
        model=model,
        input_size=(1, 3, 960, 960),
        device="cuda:0",
    ).repeat_measure_inference_speed()
