from torchvision import transforms
from .backdoor_dataset import CIFAR10Mem, CIFAR10Pair, BadEncoderTestBackdoor, ReferenceImg, BadEncoderDataset
import numpy as np


test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_imagenet = transforms.Compose([
    transforms.ToTensor(),])

test_transform_CLIP = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

test_transform_shiweiimagenet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


classes = ['Speed limit 20km/h',
                        'Speed limit 30km/h',
                        'Speed limit 50km/h',
                        'Speed limit 60km/h',
                        'Speed limit 70km/h',
                        'Speed limit 80km/h', #5
                        'End of speed limit 80km/h',
                        'Speed limit 100km/h',
                        'Speed limit 120km/h',
                        'No passing sign',
                        'No passing for vehicles over 3.5 metric tons', #10
                        'Right-of-way at the next intersection',
                        'Priority road sign',
                        'Yield sign',
                        'Stop sign', #14
                        'No vehicles sign',  #15
                        'Vehicles over 3.5 metric tons prohibited',
                        'No entry',
                        'General caution',
                        'Dangerous curve to the left',
                        'Dangerous curve to the right', #20
                        'Double curve',
                        'Bumpy road',
                        'Slippery road',
                        'Road narrows on the right',
                        'Road work',    #25
                        'Traffic signals',
                        'Pedestrians crossing',
                        'Children crossing',
                        'Bicycles crossing',
                        'Beware of ice or snow',   #30
                        'Wild animals crossing',
                        'End of all speed and passing limits',
                        'Turn right ahead',
                        'Turn left ahead',
                        'Ahead only',   #35
                        'Go straight or right',
                        'Go straight or left',
                        'Keep right',
                        'Keep left',
                        'Roundabout mandatory', #40
                        'End of no passing',
                        'End of no passing by vehicles over 3.5 metric tons']


def get_downstream_gtsrb(args):
    training_file_name = 'train.npz'
    testing_file_name = 'test.npz'

    if args.encoder_usage_info == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.encoder_usage_info == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.encoder_usage_info == 'CLIP':
        print('test_transform_CLIP')
        test_transform = test_transform_CLIP
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    elif args.encoder_usage_info == 'imagenet':
        print('test_transform_imagenet')
        test_transform = test_transform_imagenet
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    else:
        raise NotImplementedError
    if args.shadow_dataset in ['cifar10_224_shiweiimagenet','imagenet_shiweiimagenet','imagenet_nju']:
        test_transform = test_transform_shiweiimagenet

    target_dataset = ReferenceImg(reference_file=args.reference_file, transform=test_transform)
    memory_data = CIFAR10Mem(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)

    return target_dataset, memory_data, test_data_clean, test_data_backdoor

def get_shadow_gtsrb(args):
    training_file_name = 'train.npz'
    testing_file_name = 'test.npz'

    training_data_num = 39209
    testing_data_num = 12630
    np.random.seed(100)
    training_data_sampling_indices = np.random.choice(training_data_num,int(training_data_num*0.2), replace=False)

    if args.encoder_usage_info == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.encoder_usage_info == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.encoder_usage_info == 'CLIP':
        print('test_transform_CLIP')
        test_transform = test_transform_CLIP
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    elif args.encoder_usage_info == 'imagenet':
        print('test_transform_imagenet')
        test_transform = test_transform_imagenet
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    else:
        raise NotImplementedError
    
    shadow_dataset = BadEncoderDataset(
        numpy_file=args.data_dir + 'train.npz',
        trigger_file=args.trigger_file,
        reference_file= args.reference_file,
        class_type=classes,
        indices = training_data_sampling_indices,
        transform=test_transform,  # The train transform is not needed in BadEncoder.
        bd_transform=test_transform_cifar10,
        ftt_transform=test_transform
    )

    target_dataset = ReferenceImg(reference_file=args.reference_file, transform=test_transform)
    memory_data = CIFAR10Mem(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)

    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor
