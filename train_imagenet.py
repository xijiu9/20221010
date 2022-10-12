import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected in train_imagenet.')

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--twolayers_gradweight', type=str2bool, default=False, help='use two 4 bit to simulate a 8 bit')
parser.add_argument('--twolayers_gradinputt', type=str2bool, default=False, help='use two 4 bit to simulate a 8 bit')
parser.add_argument('--lsqforward', type=str2bool, default=False, help='apply LSQ')

parser.add_argument('--training-bit', type=str, default='', help='weight number of bits',
                    choices=['exact', 'qat', 'all8bit', 'only_weight', 'weight4', 'all4bit', 'forward8', 'forward4'])
parser.add_argument('--training-strategy', default='scratch', type=str, metavar='strategy',
                    choices=['scratch', 'checkpoint', 'checkpoint_from_zero', 'checkpoint_full_precision'])
parser.add_argument('--checkpoint_epoch_full_precision', type=int, default=0, help='full precision')
parser.add_argument('--clip-grad', type=float, default=10, help='clip gradient')
parser.add_argument('--amp', action='store_true', help='Run model AMP (automatic mixed precision) mode.')

parser.add_argument('--arch', type=str, default='resnet50', help='clip gradient to 0.01(CIFAR)')
parser.add_argument('--gpu-num', type=int, default=8, help='clip gradient to 0.01(CIFAR)')
parser.add_argument('--lr', type=float, default=1.024, help='clip gradient to 0.01(CIFAR)')
parser.add_argument('--weight-decay', '--wd', default=3.0517578125e-05, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--batch-size', type=int, default=128, help='clip gradient to 0.01(CIFAR)')
parser.add_argument('--optimizer-batch-size', type=int, default=1024, help='clip gradient to 0.01(CIFAR)')
parser.add_argument('--warmup', default=4, type=int, metavar='E', help='number of warmup epochs')
args = parser.parse_args()

assert args.gpu_num * args.batch_size == 1000 * args.lr
assert args.gpu_num * args.batch_size == args.optimizer_batch_size

arg = " -c quantize --qa=True --qw=True --qg=True"

if args.training_bit == 'all8bit':
    bbits, bwbits, awbits = 8, 8, 8
elif args.training_bit == 'exact':
    arg = ''
    bbits, bwbits, awbits = 0, 0, 0
elif args.training_bit == 'qat':
    bbits, bwbits, awbits = 0, 0, 8
    arg = "-c quantize --qa=True --qw=True --qg=False"
elif args.training_bit == 'only_weight':
    bbits, bwbits, awbits = 8, 4, 4
elif args.training_bit == 'weight4':
    bbits, bwbits, awbits = 8, 4, 8
elif args.training_bit == 'all4bit':
    bbits, bwbits, awbits = 4, 4, 4
elif args.training_bit == 'forward8':
    bbits, bwbits, awbits = 4, 4, 8
elif args.training_bit == 'forward4':
    bbits, bwbits, awbits = 8, 8, 4
else:
    bbits, bwbits, awbits = 0, 0, 0

if args.twolayers_gradweight:
    assert bwbits == 4
if args.twolayers_gradinputt:
    assert bbits == 4

if args.twolayers_gradweight and args.twolayers_gradinputt:
    method = 'twolayer'
elif args.twolayers_gradweight and not args.twolayers_gradinputt:
    method = 'twolayer_weightonly'
elif not args.twolayers_gradweight and not args.twolayers_gradinputt:
    method = args.training_bit

if args.training_strategy == 'checkpoint' or args.training_strategy == 'checkpoint_from_zero':
    model = ''
elif args.training_strategy == 'checkpoint_full_precision':
    model = ''
else:
    model = 'pass'

if args.amp:
    amp_control = '--amp --static-loss-scale 128'
else:
    amp_control = ''

os.system("python ./multiproc.py --nnodes 1 --node_rank 0 --master_addr '127.0.0.1' \
            --nproc_per_node {} ./main.py --arch {} --gather-checkpoints \
            --batch-size {} --lr {} --optimizer-batch-size {} --resume {}\
            --warmup {} {}  /data/LargeData/Large/ImageNet --workspace ./results/imagenet/{}/models \
            {} --print-freq 400 --clip-grad {} \
            --bbits {} --bwbits {} --abits {} --wbits {} --weight-decay {} --lsqforward {} \
            --twolayers-gradweight {} --twolayers-gradinputt {} --clip-grad {}"
            .format(args.gpu_num, args.arch,
                    args.batch_size, args.lr, args.optimizer_batch_size, model,
                    args.warmup, arg, args.training_bit,
                    amp_control, args.clip_grad,
                    bbits, bwbits, awbits, awbits, args.weight_decay, args.lsqforward,
                    args.twolayers_gradweight, args.twolayers_gradinputt, args.clip_grad))