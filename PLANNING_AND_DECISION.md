Our project was start on later November, we trying to find some project about Federated Learning and 
one of the TA suggest us to do some FL with TI-SLAM. 

Then we start to read the paper and try to understand the code. Since we learn about
Fl with Flower framework in the lab so we decided to use Flower for this project.

But Flower didn't support TF1, and TI-SLAM was originally written in TF1.
So we spent quite some time to convert the code to TF2, makesure it correct 
and do the cross validation between TF1 and TF2.

Then we start to implement the FL part, we use the code from Flower simulation tutorial
and lab3 as references. 

We was planning on experiment with different FL global model update strategy, 
differnt clients/round/epcohs setting and different dataset domian, as TI-SLAM paper 
mentioned it lack of ability to generalize across environment. Our initial plan was
ambitious, but we didn't expect that training takes huge time. Single FL subnetwork
training takes 6 hours and sadly 2080Ti is the best GPU we have. We also rent a Google Cloud
with some P4 GPU but it doesn't help much.

In the end we only able to finish the experiment with subnetwork FL and its result on
SLAM performance. Very sad that we didn't enough computing power to finish the experiment. Otherwise
it will be a very interesting project to discover the FL performance on SLAM problem. 
We expect a huge gain on cross domain issue.