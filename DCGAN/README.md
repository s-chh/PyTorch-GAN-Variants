# Pytorch-DCGAN
Pytorch implementation of [DCGAN](https://arxiv.org/abs/1511.06434) for generating 64x64 images.

### LSUN Dataset
To download LSUN dataset follow the steps at [https://github.com/fyu/lsun](https://github.com/fyu/lsun)

<br>
## Run commands (also available in <a href="./scripts.sh">scripts.sh</a>): <br>

<table>
  <tr>
    <th>Dataset</th>
    <th>Run command</th>
  </tr>
  <tr>
    <td>MNIST</td>
    <td>python main.py --dataset mnist --n_channels 1 --epochs 25</td>
  </tr>
  <tr>
    <td>CelebA</td>
    <td>python main.py --dataset celeba</td>
  </tr>
  <tr>
    <td>LSUN Church</td>
    <td>python main.py --dataset lsun_church</td>
  </tr>
  <tr>
    <td>LSUN Bedroom</td>
    <td>python main.py --dataset lsun_bedroom</td>
  </tr>
</table>


## Generated Samples
#### LSUN-Bedroom
<img src="./Results/LSUN_Bedroom.png" width="700"></img>
#### LSUN-Church
<img src="./Results/LSUN_Church.png" width="700"></img>
#### MNIST
<img src="./Results/MNIST.png" width="700"></img>
#### CelebA
<img src="./Results/CelebA.png" width="700"></img>
