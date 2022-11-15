import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from selfholo import *
from imageio import imread

device = torch.device('cuda')
checkpoint = torch.load("./checkpoints/green.pth")
channel = 1

self_holo = selfholo().to(device)
self_holo.load_state_dict(checkpoint)
self_holo.eval()
target_path = ("./dataset/example/")
recon_path = ("./recon")
holo_path = ("./recon/holo")

for i in range(1):

    #img = ['bunny', 'castle']
    #depth_char = 'b'
    img = ['bbb']
    depth_char = 'depth'

    img_name = combine(img[i], '.png')
    depth_name = combine(depth_char, '.png')
    all_name = combine(img[i], depth_char)
    img_path = os.path.join(target_path, img_name)
    print(img_path)
    depth_path = os.path.join(target_path, depth_name)

    img = imread(img_path)
    if len(img.shape) < 3:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    img = img[..., channel, np.newaxis]
    im = im2float(img, dtype=np.float32)  # convert to double, max 1
    low_val = im <= 0.04045
    im[low_val] = 25 / 323 * im[low_val]
    im[np.logical_not(low_val)] = ((200 * im[np.logical_not(low_val)] + 11)
                                   / 211) ** (12 / 5)
    amp = np.sqrt(im)  # to amplitude
    amp = np.transpose(amp, axes=(2, 0, 1))
    amp = resize_keep_aspect(amp, [880, 880])
    target = np.reshape(amp, (880, 880))
    amp = np.reshape(amp, (1, 1, 880, 880))
    amp = torch.from_numpy(amp)
    amp = amp.to(device)

    depth_map = imread(depth_path)
    if len(depth_map.shape) < 3:
        depth_map = np.repeat(depth_map[:, :, np.newaxis], 3, axis=2)
    depth_map = depth_map[..., 1, np.newaxis]
    depth_map = im2float(depth_map, dtype=np.float32)
    depth_map = np.transpose(depth_map, axes=(2, 0, 1))
    depth_map = resize_keep_aspect(depth_map, [880, 880])
    depth_map = np.reshape(depth_map, (1, 1, 880, 880))
    depth_map = torch.from_numpy(depth_map)
    depth_map = depth_map.to(device)

    input = torch.cat([amp, depth_map], dim=-3)
    source = pad_image(input, [1072, 1072], padval=0, stacked_complex=False)
    resoution = [880, 880]

    if (depth_char == 'b'):

        holo, slm_amp, recon_field = self_holo(source, 0)
        recon = crop_image(recon_field, resoution, pytorch=True, stacked_complex=False)
        recon = recon.abs()
        recon = recon.squeeze().cpu().detach().numpy()
        psnr_val, ssim_val = get_psnr_ssim(recon, target, multichannel=False)
        print(" PSNR:{}, SSIM:{}".format(psnr_val, ssim_val))
        # save reconstructed image in srgb domain
        recon = srgb_lin2gamma(np.clip(recon ** 2, 0.0, 1.0))
        recon = (recon - recon.min()) / (recon.max() - recon.min())
        img_name = combine(all_name, channel, '.png')
        path = os.path.join(recon_path, img_name)
        imwrite(recon, path)

    elif (depth_char == 'w'):

        holo, slm_amp, recon_field = self_holo(source, 2)
        recon = crop_image(recon_field, resoution, pytorch=True, stacked_complex=False)
        recon = recon.abs()
        recon = recon.squeeze().cpu().detach().numpy()
        psnr_val, ssim_val = get_psnr_ssim(recon, target, multichannel=False)
        print("PSNR:{}, SSIM:{}".format(psnr_val, ssim_val))
        # save reconstructed image in srgb domain
        recon = srgb_lin2gamma(np.clip(recon ** 2, 0.0, 1.0))
        img_name = combine(all_name, channel, '.png')
        path = os.path.join(recon_path, img_name)
        imwrite(recon, path)

    else:
        image_name = ['n', 'm', 'f']
        for j in range(len(image_name)):

            holo, slm_amp, recon_field = self_holo(source, j)
            recon = crop_image(recon_field, resoution, pytorch=True, stacked_complex=False)
            recon = recon.abs()
            recon = recon.squeeze().cpu().detach().numpy()
            psnr_val, ssim_val = get_psnr_ssim(recon, target, multichannel=False)
            print("In:{}, PSNR:{}, SSIM:{}".format(image_name[j], psnr_val, ssim_val))
            # save reconstructed image in srgb domain
            recon = srgb_lin2gamma(np.clip(recon ** 2, 0.0, 1.0))
            recon = (recon - recon.min())/(recon.max()-recon.min())
            img_name = combine(all_name, channel, image_name[j], '.png')
            path = os.path.join(recon_path, img_name)
            imwrite(recon, path)

    # save hologram
    holo = torch.squeeze(holo)
    holo = holo.cpu().detach().numpy()
    output_phase = (holo + np.pi) / (2 * np.pi)
    holo_name = combine(all_name, channel, '.png')
    con_holo_path = os.path.join(holo_path, holo_name)
    imwrite(output_phase, con_holo_path)








