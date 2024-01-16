Our project was started in late November, we found some projects about Federated Learning and 
one of the TAs suggested us to do some FL with TI-SLAM. 

Then we start to read the paper and try to understand the code. Since we learn about
Fl with Flower framework in the lab so we decided to use Flower for this project.

When testing the code of TI-SLAM and attempting to implement some opening experiments, we noticed the 
problem of incompatibility within libraries, especially when implementing the federated learning part 
using flowers. Flower doesn't support TF1, but TI-SLAM was originally written in TF1.

So we summarised the challenges encountered at that stage and decided to convert the entire TI-SLAM from 
TF1 to TF2. This took a considerable amount of effort. To ensure things were working, we cross-validated
the results of two TI-SLAM of two versions of TF and ensured that the code worked as before. 

Then we start to implement the FL part, we use the code from Flower simulation tutorial
and lab3 as references. 

We planned to conduct experiments with different FL global model update strategies, 
different clients/round/epochs settings and different dataset domains, as the TI-SLAM paper 
mentioned its lack of ability to generalize across environments. The experiments were run on Hantao's
computer and Haochen's virtual machine on Google Cloud. The size of the dataset is more than 400 Gigabytes.
We encountered problems with the Docker environment suggested by the original paper when running experiments 
and switched to Conda in the end. 

At the early stage of training, we also rented a Google Cloud with a Nvidia P4 GPU but it didn't help much.
The platform later increased our quota to 2 P4 GPUs, but this was still insufficient for 
conducting all prospective experiments at full scale. The FL client crashed multiple times during training
due to exhausted resources, and when verifying the error logs, we found that Ray did not perform 
garbage collection which caused unnecessary occupancy of VRAM, and Haotao fixed this issue. But this was 
not enough, as a single FL subnetwork training takes 6 hours and sadly 2080Ti is the best GPU we have 
(Hantao's computer). 

Though facing obstacles, we finished the experiments with subnetwork FL and its result on SLAM performance.
We are happy to see FL worked and produced meaningful results. We also think that more work on this
project can be done later for more findings about applying FL to SLAM, as the combination 
of FL and SLAM could have some meaningful user scenarios.

Thanks!

Haochen Liu and Haotao Zhong
