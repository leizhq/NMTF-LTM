# NMTF-LTM: Towards an Alignment of Semantics for Lifelong Topic Modeling



## 1 Our environment

For our experiments, we run the codes on the OS of Linux, with Python 3.8.



## 2 How to run the codes

### 2.1 NMTF-LTM

Firstly, you need to install the necessary packages.

```shell
pip install -r requirements_NMTF-LTM.txt
```

Then, you can use the shell prepared by us to perform model running and evaluation on a Demo dataset which contains the first 5 chunks of the 20News dataset in the paper.

```shell
bash run_and_evaluate_NMTF-LTM.sh Demo_NMTF-LTM
```

Note that the last argument is the experiment id in "experiment.ini" where you can specify the model, dataset, and parameters.

### 2.2 PNMTF-LTM

For PNMTF-LTM, you first need to deploy an MPI environment, for example, MPICH or OpenMPI. Also, it is recommended to run PNMTF-LTM on a high-performance computing cluster or a supercomputer to achieve  observable speedup.

Then, you need to install the necessary packages. 

```shell
pip install -r requirements_PNMTF-LTM.txt
```

The only difference between "requirements_PNMTF-LTM.txt" and "requirements_NMTF-LTM.txt" is the requirement of mpi4py package.

Finally, you can use the shell prepared by us to perform model running and evaluation on a Demo dataset which contains the first 5 chunks of the 20News dataset in the paper.

```
bash run_and_evaluate_PNMTF-LTM.sh Demo_PNMTF-LTM
```

Note that the last argument is the experiment id in "experiment.ini" where you can specify the model, dataset, and parameters.