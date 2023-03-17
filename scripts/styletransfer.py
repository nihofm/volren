# external imports
import PIL
import time
import torch
import argparse
import torchvision

# -----------------------------------------------------------
# cmd line settings

cmdline = argparse.ArgumentParser(description='Neural style transfer')
# required args
cmdline.add_argument('content_path', help='path to content image')
cmdline.add_argument('style_path', help='path to style image')
# optional args
cmdline.add_argument('-e', '--epochs', type=int, default=1000, help='number of epochs to train')
cmdline.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='adam hyperparameter: learning rate')
cmdline.add_argument('-b1', '--beta1', type=float, default=0.9, help='adam hyperparameter: beta1')
cmdline.add_argument('-b2', '--beta2', type=float, default=0.999, help='adam hyperparameter: beta2')
cmdline.add_argument('-c', '--clip', type=float, default=1, help='value for gradient value/norm clipping')
cmdline.add_argument('-g', '--gamma', type=float, default=0.999, help='step lr gamma parameter')
cmdline.add_argument('--save_epochs', type=int, default=50, help='after which number of epochs to save image')
cmdline.add_argument('--image_size', type=int, default=512, help='output image size')
cmdline.add_argument('--content', type=float, default=1, help='content loss weight')
cmdline.add_argument('--style', type=float, default=3000, help='style loss weight')
cmdline.add_argument('--dropout', type=float, default=0.0, help='gradient dropout probability')
cmdline.add_argument('--noise', action="store_true", help="start from random noise")

# -----------------------------------------------------------
# Helper funcs and modules

# L2 loss
def l2(img1, img2):
    return torch.nn.functional.mse_loss(img1, img2)

# SMAPE loss
def smape(img1, img2):
    return torch.mean(torch.abs(img1 - img2) / (torch.abs(img1) + torch.abs(img2) + 0.1))

# compute gram matrix
def gram_matrix(img):
    b, c, h, w = img.size()
    tmp = img.view(b*c, h*w)
    return torch.mm(tmp, tmp.t()) / (b*c*h*w)

# VGG style loss module
class VGGStyleLoss(torch.nn.Module):
    def __init__(self, content_weight, style_weight, dropout):
        super(VGGStyleLoss, self).__init__()
        self.model = torchvision.models.vgg11(weights=torchvision.models.vgg.VGG11_Weights.DEFAULT).features
        self.model.eval()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.dropout = dropout

    def get_features(self, x):
        features = []
        for layer in self.model.children():
            x = layer(x)
            if isinstance(layer, torch.nn.ReLU):
                features.append(torch.nn.functional.dropout(x, self.dropout))
        return features

    def forward(self, image, content, style):
        self.model.to(image.get_device())
        mask = torch.where(content > 0, 1, 0)
        features = self.get_features(torch.cat([mask * image, content, style]))
        feature_loss, style_loss = 0.0, 0.0
        for i in range(len(features)):
            f_input, f_content, f_style = features[i][0:1], features[i][1:2], features[i][2:3]
            feature_loss += self.content_weight * l2(f_input, f_content)
            style_loss += self.style_weight * smape(gram_matrix(f_input), gram_matrix(f_style))
        return (feature_loss + style_loss) / len(features)

# -----------------------------------------------------------
# MAIN

if __name__ == "__main__":

    # parse command line
    args = cmdline.parse_args()
    print('----------')
    print('args:')
    for key in vars(args):
        print(key, ":", vars(args)[key])
    print('----------')

    # check for GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'Num GPUs: {torch.cuda.device_count()}')

    def image_loader(filename):
        return torchvision.transforms.ToTensor()(PIL.Image.open(filename))[0:3].unsqueeze(0).to(device, torch.float)

    # load content and style images
    content = image_loader(args.content_path)
    style = image_loader(args.style_path)
    content = torch.nn.functional.interpolate(content, scale_factor=min(1.0, args.image_size / max(content.size())), recompute_scale_factor=False, mode='bicubic')
    style = torch.nn.functional.interpolate(style, size=content.size()[-2:], mode='bicubic')
    # setup image to optimize
    image = torch.rand(content.size(), device=device) * torch.where(content > 0, 1, 0) if args.noise else content.clone()

    # setup optimizer, scheduler and loss function
    optimizer = torch.optim.Adam([image.requires_grad_()], lr=args.learning_rate, betas=(args.beta1, args.beta2))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    loss_fn = VGGStyleLoss(content_weight=args.content, style_weight=args.style, dropout=max(0.0, min(args.dropout, 1.0)))

    # run training
    print(f'Transferring style from {args.style_path} to {args.content_path} ({style.size()[-1]}x{style.size()[-2]}) for {args.epochs} epochs.')

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        optimizer.zero_grad()
        loss = loss_fn(image, content, style)
        loss.backward()
        max_grad = torch.max(image.grad.data.abs())
        torch.nn.utils.clip_grad_value_([image], args.clip)
        optimizer.step()
        scheduler.step()
        image.data.clamp_(0, 1)
        # save image?
        if epoch % args.save_epochs == 0:
            torchvision.utils.save_image(image, f'styletransfer.png', nrow=1)
            torchvision.utils.save_image(torch.cat((content, image, style), dim=-1), f'p_style.jpg', nrow=1)
        torch.cuda.synchronize()
        end = time.time()
        print(f"Train epoch {epoch:04}:",
                f"LR: {optimizer.param_groups[0]['lr']:0.6f},",
                f"loss: {loss:3.6f},",
                f"max grad: {max_grad:4.4f},",
                f"time: {(end - start)*1000:.0f}ms (remaining: ~{(end - start) * (args.epochs - epoch) / 60:.1f}min)", end='\r')
    print('')

    # finished, save image
    torchvision.utils.save_image(image, f'styletransfer.png', nrow=1)
