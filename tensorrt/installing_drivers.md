# Installing GPU drivers/CUDA

You don't necessarily need to do this, I think the docker container works with just the driver and without system-level CUDA, but I did it anyway as part of earlier testing, plus it pulls in a newer driver version than the general release one.

These instructions worked on a (mostly) clean **Ubuntu 20.04**.

If you have another driver, first uninstall it:
```
sudo apt remove nvidia-driver-455
sudo apt autoremove
reboot
```
This is where you're supposed to disable/remove the nouveau driver if you have it, but I already had a nvidia driver.

**Warning:** This is Linux GPU stuff so things will break!

In GRUB, press `e` to edit the boot command and in the line with `quiet splash`, replace that with `nomodeset`.
 
Press `F10` to boot with the edited command, and you should get the graphical login screen again :)


Now install cudatoolkit:

## 11.3:
```
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
sudo sh cuda_11.3.1_465.19.01_linux.run
```

## 10.2:
```
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
```

To test that it's working use `nvidia-smi`:
```
(base) talmo@talmo-desktop-ubuntu:~$ nvidia-smi
Sat May 22 20:25:54 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.19.01    Driver Version: 465.19.01    CUDA Version: 11.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA TITAN RTX    Off  | 00000000:01:00.0  On |                  N/A |
| 41%   39C    P0    59W / 280W |   1999MiB / 24186MiB |      2%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

```
