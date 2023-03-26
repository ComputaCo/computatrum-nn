# Computatrum

The aim of this project is to develop an
- AI system that can interact with a computer
- following human language instructions
- or independantly interacting with the computer
- and that can optimize and deploy its own code
- all subject to financial and technical constraints of an average programmer.

## Motivation

Machine learning has a general recipe for developing increasingly advanced systems: we identify and optimize various ‘components of intelligence' (such as datasets, environments, training paradigms, objectives, and architecture designs) and then incrementally integrate them under an iterative dev-test cycle. We do this while minimizing the amount of information movement required (how much data we have to collect, how long we train the model, and how much cognitive load our own mind handles through design and experimentation). The human mind is a powerful at solving machine, but **even at the iteration speed of an expert research team with high end equipment, there are just too many accidental and essential complexities, and not enough automation driving ML evolution from end-to-end**. Now in deep learning, the phrase “end-to-end” means you don’t have to hand-engineer a bunch of parameters, but state-of-the-art deep neural networks today are still relatively hardwired when you consider that they have to be spoon fed datasets, told when to wake up, train, and die, and they train under an economic fitness landscape that they have no direct awareness of. That is why they still employ us as machine learning engineers to tune the remaining 100 hyperparameters as well as the uncountably many unknown parameters. **If machine learning is going to approach and surpass the rate-limiting bar of human research and development, we need to liberate as many aspects of the ML development cycle as possible to autonomous control.**

This motivates developing systems that genuinely propagate feedback from ‘end-to-end’, hunt for their own data, pursues their own cultivated intrinsic motivations, act as their own economic entity subject to the same financial and technological constraints as an independant human engineer, and are able to research and develop state-of-the-art ML systems -- including improvements of themselves. I am not just reformulating autoML, unsupervised/reward-free/intrinsically-motivated reinforcement learning, or some evolutionary AI-generating formal algorithm. **I propose developing a fully autonomous, open-ended machine learning system that which interacts using standard peripherals connected to a virtual machine running Ubuntu with Internet access to interact with robots, research sites, and its own software and compute resources.** I call this system Computatrum. Following is a description of the system’s architecture and the current state of development.

## Design

Perception [Action] [Online learning] [Offline learning]

## Setup

```bash
apt-get install tesseract-ocr libtesseract-dev libleptonica-dev pkg-config

poetry install
```