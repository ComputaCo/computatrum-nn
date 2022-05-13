---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
---

<!--
This content is kept in sync with `theory.ipynb` via `jupytext`. Run

  ```bash
  pip install jupytext

  # to convert the md into ipynb format (if no .ipynb file exists)
  jupytext --set-formats ipynb,md --sync task-guided-computer-interaction.md

  # to convert the ipynb into md format (if no .md file exists)
  jupytext --set-formats ipynb,md --sync task-guided-computer-interaction.ipynb
  ```

to link the markdown and ipynb versions of this file. Once linked, run

  ```bash
  jupytext --sync task-guided-computer-interaction.ipynb
  ```

to update either file with the other's changes.
-->

# Task Guided Computer Interaction

## Introduction

Introduction

Motivation

Overview

## Automating Computer interaction

Main idea

Explain several categories of tasks (including sections of user manuals and YouTube videos)

Common features shared by many or all tasks include:

- Being able to operate at varying levels of abstraction simultaneously.
- Another common feature of all tasks is...

## Environment

Formal model

Code: the code's on my github (footnote with link). Explain how to use it.

```python
example using random action space samples
```

Animated GIF output

And if I wanted to do action Y,

```python
example of performing a preprogrammed action sequence
```

Animated GIF output

Conclude this section and lead into the next two: data and evaluation.

## Demonstrations

Explain that there is a need for a large number of demonstrations

### Collecting demonstrations

I collect demonstrations using ... Explain how. Show code/script:

```python
# collect demonstrations
```

Using this methodology, I collected a dataset of diverse GUI interactions spanning ENUMERATE. Details on this dataset. I uploaded it to WHERE. It consist of the following demonstrations:

```python
# show how to download dataset
# display dataset
```

(dataset dataframe)

### Augmented demonstrations

Data augmentations. Explain the concept and justify. I use these augmentations:

- mouse/keyboard jerky/smooth/fast/slow in-between press/release
- different display sizes, observation-action update rates
- different window themes, accessibility features enabled/disabled, and other visual variations
- Different language prompt variations
- In some cases, resize the window
- general augmentation approaches: frame skipping, dropping, noise

These augmentations require the demonstrations to be performed in a deterministic environment, OTHER CONSTRAINTS, however using them, I am able to expand the dataset from X demonstrations to Y total demonstrations -- a ZZZ% increase.

Show code/script:

```python
# augment all data in original dataset
# print demonstration count before and after
# select a few demonstrations to display
```

(outputs)

### Synthetic demonstrations

Explain the concept and justify.

The synthetic task curriculum includes:

- low level motor instructions:
- automated form filling: the form is auto-generated and must be filled and submitted with specified data.
- TODO

Also, all of the above is performed with instructions given

- in a separate "task description" modality
- before computer interaction begins as a webpage (agent must minimize/leave browser to begin executing task)
- before computer interaction begins as a video tutorial (agent must minimize/leave browser to begin executing task)

See `gui-bot-language.md` for more details. (TODO. Move to separate repository)

Give an example of using this tool:

```python

```

(Show some generated tasks and their execution by the bot)

## General pretraining

General pretraining employs a mixture of autoregressive modeling with supervised learning and semantic alignment with proximal policy optimization. Agents train with and without assistance which may be provided in the form of observing a full observation-action trajectory demonstration, watching an on-screen video tutorial, reading an on-screen web-page tutorial before starting the episode, or ad-hoc teacher forcing if their trajectory derails from the demonstration. Hyperparameter optimization dynamically determines task domain sample weighting, task complexity, TODO, and optimizer parameters.

(Make a hidden code cell show all these training configurations)
(show include and exclude rules/combinations)

TODO: consider the distinction between unlabeled and labeled demonstrations
TODO: clarify AR as input reconstruction AND/OR input prediction AND/OR action prediction (for demos only) etc TODO

```text
( 
  autoregressive modeling optimized by supervised learning on C(input reconstruction, outputs, input predictions),
  PPO with reward network (which simultaneously learns from
    {demos (+),demos (+) and collected_traj (-)})
)
where the inputs come from {live interaction, demonstration}
with {no,video,webpage,ad-hoc teacher forcing} assistance.

Exceptions:
- no demonstration and ad-hoc teacher forcing.
- no demonstration and PPO

Also unsupervised exploration using {input reconstruction AND/OR input prediction} on {input,output,all} modalities with data collected from live interaction with no assistance.

Also {input reconstruction AND/OR prediction} on video tutorials from YouTube 
```

### Self-supervised approaches

Introduce input reconstruction and autoregressive modeling

Explain why imoprt: a lot of data doesn't have to be labeled

Show snippets of loss function for ReconstructionEnvironment and AutoregressiveEnvironment.

### Supervised

Introduce

Show python example

### Semantic alignment

(This section should go after the synthetic dmeonstrations section because I need to explain how I train the action-language model)

The main idea. make a text summary of the state/trajectory. Compare that summary against an expected summary in semantic embedding space. Minimize the distance between the two.

Briefly highlight shortcomings of naive approaches (static or dynamic-time classification) to semantic alignment.

- These are 'quick-trick', one-off, brittle hand-engineered approaches that work -- but fail to generalize.
- The problem is that most tasks are too complex to parametrize and there isn't enough data.
- These are naive by themselves, but not necessarily a bad idea.
- like Imitation learning and Task classification

Introduce a more general, principled, and preferred way to evaluate task description alignment and explain why.

Static vision-language analysis

Example with CLIP (or whatever vision-language model I plan on using)

```python
# Re-use a frame from the GIF input from previously
```

(show frame and text summary)

Dynamic task estimation

Example with another language model

```python
# Re-use the GIF input from previously
```

(show animation and text summary)

Explain how these approaches can be formed into a reward function.

Algorithm.

Code snippets of reward function from computatrum repo.

Demonstrate how to use the reward function.

```python
# Re-use the GIF input from previously
```

(show inputs and outputs)

Convince the reader that this reward function is hard to game. Try to perform the task manually and see if it's possible to get a high reward without performing the objective.

```python
# show inputs (myself) and outputs (my score)
```

Being able to directly measure similarity between state/trajectory and task description allows us to train on a far more diverse array of tasks than possible with only the above. Explain the limitations of CLIP: it's mainly good with natural images. I make a curriculum that utiliizes these features. URL. Here is a subset:

```python
# from each class, sample 10 example task descriptions
```

(Cell output:
motor-level tasks: (LMB down, press A)

- task1
- task2
- task3
...

low-level tasks: (type "hello", click the blue triangle)

- task1
- task2
- task3
...

UI understanding (check the checkbox, scroll the scrollbar, open a window)

- task1
- task2
- task3
...

App understanding (open email, go to "google.com", copy the files in /tmp to /home/jacob/backup)

- task1
- task2
- task3
...

Language skills (email Phil that I'll be out tomorrow)

- task1
- task2
- task3
...

Visual skills (less general than the others; draw a picture of a dog)

- task1
- task2
- task3
...

Instruction following (follow along to the YouTube tutorial, follow these steps to install tensorflow)

- task1
- task2
- task3
...

)

Then training an agent to follow tasks is simply:

```python
# build languageAlignmentEnv
# make a simple policy
# train it
# evaluate it
# show best example
```

Since CLIP possesses a large amount of world knowledge burned into its parameters, we can even ask the policy to do things like draw natural images and get a human-like behavior response:

```python
# command the trained policy from above
# to draw a picture of a cat
# start out with the paint program open
```

Super cool!

### Ad-hoc teacher forcing

```python
Helper(Env, dataclass):
  env: Env
  demo: Trajectory
  dist: Callable
  threshold: float

  def step(self, action, **kwargs):
    if dist(prev_obs, demo_obs) > threshold:
      action = demo_action
    self.env.step(action, **kwargs)

bot = GUIBot(...)
env = ComputerEnv(...)
demo = executer.collect(env, bot)
def mouse_movement_only_dist(demo, policy):
  # if anything besides the wheel or position is different,
  # make distance > threshold
  # otherwise, return the raw wheel and/or movement euclidean distance
  pass
threshold = 10
env = Helper(env, demo, mouse_movement_only_dist, threshol)
# demonstrate
```

### Curriculum learning

Methodology: go from easier to harder based on how the learner is progressing. Advantages and disadvantages. In this case, the advantages seem to outweigh the disadvantages. This is the curriculum:

I need to explain that the final evaluator is a composition of several evaluators since there is just too much complexity to unify everything into a common framework. This means I will need to re-order the sections and make training and evaluation a common section.

Introduce the `CurriculumEnv`.

```python
# all of the above code should have already
# defined the environments that will be needed
env = CurriculumEnv(TODO)
```

## Delete

### Environment-Plumbing

Because I employ rather than a single training methodology, the explicit reward function that Computatrum trains on is heterogeneous. Explain how I couldn't get this to fit inside the standard RL/SL framework and how this motivated me to develop the Artificial Experience framework. Introduce it and give lots of examples.

Introduce the artificial experience and its abstractions. Give a couple of examples building an environment for computer interaction.

### Collecting demonstrations

I collect demonstrations using ... Explain how. Show code/script:

```python
# collect demonstrations
```

Using this methodology, I collected a dataset of diverse GUI interactions spanning ENUMERATE. Details on this dataset. I uploaded it to WHERE. It consist of the following demonstrations:

```python
# show how to download dataset
# display dataset
```

(dataset dataframe)

I can prepare this dataset as an environment for teacher forcing like so:

```python
# build a dataset environment with the above dataset
# show it being used
# (inside of a gradient tape to optimize loss on each iteration)
```

### Augmented demonstrations

Data augmentations. Explain the concept and justify. I use these augmentations:

- mouse/keyboard jerky/smooth/fast/slow in-between press/release
- different display sizes, observation-action update rates
- different window themes, accessibility features enabled/disabled, and other visual variations
- Different language prompt variations
- In some cases, resize the window
- general augmentation approaches: frame skipping, dropping, noise

These augmentations require the demonstrations to be performed in a deterministic environment, OTHER CONSTRAINTS, however using them, I am able to expand the dataset from X demonstrations to Y total demonstrations -- a ZZZ% increase.

Show code/script:

```python
# augment all data in original dataset
# print demonstration count before and after
# select a few demonstrations to display
```

(outputs)

### Synthetic demonstrations

Explain the concept and justify.

The synthetic task curriculum includes:

- low level motor instructions:
- automated form filling: the form is auto-generated and must be filled and submitted with specified data.

-

Also, all of the above is performed with instructions given

- in a separate "task description" modality
- before computer interaction begins as a webpage (agent must minimize/leave browser to begin executing task)
- before computer interaction begins as a video tutorial (agent must minimize/leave browser to begin executing task)

See `gui-bot-language.md` for more details. (TODO. Move to separate repository)

Give an example of using this tool:

```python

```

(Show some generated tasks and their execution by the bot)

Give an example of wrapping this tool into the environment demonstration forcing wrapper:

```python

```

### Input reconstruction and autoregressive modeling

All of the aforementioned environments represent a supervised approach to training. However we need to develop agents that can perform tasks which were never done before. This is where intrinsic reward objectives become useful. Explain input reconstruction and autoregression.

Show snippets of loss function for ReconstructionEnvironment and AutoregressiveEnvironment.

### Semantic alignment

The main idea. make a text summary of the state/trajectory. Compare that summary against an expected summary in semantic embedding space. Minimize the distance between the two.

Briefly highlight shortcomings of naive approaches.

- These are 'quick-trick', one-off, brittle hand-engineered approaches that work -- but fail to generalize.
- The problem is that most tasks are too complex to parametrize and there isn't enough data.
- These are naive by themselves, but not necessarily a bad idea.
- like Imitation learning and Task classification

Introduce a more general, principled, and preferred way to evaluate task description alignment and explain why.

Static vision-language analysis

Example with CLIP (or whatever vision-language model I plan on using)

```python
# Re-use a frame from the GIF input from previously
```

(show frame and text summary)

Dynamic task estimation

Example with another language model

```python
# Re-use the GIF input from previously
```

(show animation and text summary)

Explain how these approaches can be formed into a reward function.

Algorithm.

Code snippets of reward function from computatrum repo.

Demonstrate how to use the reward function.

```python
# Re-use the GIF input from previously
```

(show inputs and outputs)

Convince the reader that this reward function is hard to game. Try to perform the task manually and see if it's possible to get a high reward without performing the objective.

```python
# show inputs (myself) and outputs (my score)
```

Being able to directly measure similarity between state/trajectory and task description allows us to train on a far more diverse array of tasks than possible with only the above. Explain the limitations of CLIP: it's mainly good with natural images. I make a curriculum that utiliizes these features. URL. Here is a subset:

```python
# from each class, sample 10 example task descriptions
```

(Cell output:
motor-level tasks: (LMB down, press A)

- task1
- task2
- task3
...

low-level tasks: (type "hello", click the blue triangle)

- task1
- task2
- task3
...

UI understanding (check the checkbox, scroll the scrollbar, open a window)

- task1
- task2
- task3
...

App understanding (open email, go to "google.com", copy the files in /tmp to /home/jacob/backup)

- task1
- task2
- task3
...

Language skills (email Phil that I'll be out tomorrow)

- task1
- task2
- task3
...

Visual skills (less general than the others; draw a picture of a dog)

- task1
- task2
- task3
...

Instruction following (follow along to the YouTube tutorial, follow these steps to install tensorflow)

- task1
- task2
- task3
...

)

Then training an agent to follow tasks is simply:

```python
# build languageAlignmentEnv
# make a simple policy
# train it
# evaluate it
# show best example
```

Since CLIP possesses a large amount of world knowledge burned into its parameters, we can even ask the policy to do things like draw natural images and get a human-like behavior response:

```python
# command the trained policy from above
# to draw a picture of a cat
# start out with the paint program open
```

Super cool!

### Curriculum learning

Methodology: go from easier to harder based on how the learner is progressing. Advantages and disadvantages. In this case, the advantages seem to outweigh the disadvantages. This is the curriculum:

I need to explain that the final evaluator is a composition of several evaluators since there is just too much complexity to unify everything into a common framework. This means I will need to re-order the sections and make training and evaluation a common section.

Introduce the `CurriculumEnv`.

```python
# all of the above code should have already
# defined the environments that will be needed
env = CurriculumEnv(TODO)
```

## Network Architecture

Design philosophy and criterion for policy architecture(s).

I'd be willing to bet that if neuroscientists analyzed me and my millennial-peer's brains, they find a structure in the homunculus or SMA just dedicated to the mouse. But the intelligence should extend far beyond pixel-level representations. (Picture of butterfly, dog, baby, man, and manager with different perspectives/abstractions on the man's actions.)

Choices made.

Picture of policy architecture(s).

## Putting it all together

Figure showing entire architecture.

Explain overall architecture.

Explain details.

Provide some justification for design. Link to my separate posts on various design criteria.

## General Pre-training

Introduce hardware and software specifics. Date, time, server location, etc.

Show tensorboard graphs of loss over time

Final results. Link to checkpoints.

## Experiments

TODO: I'm not sure if there needs to be an experiments section

Each experiment should include:

- user demonstration
- invocation code
- quantitative analysis (metrics)
- at least one raw animation
- other qualitative analysis (visualizations)
- critical summary

## Use Cases

This section expands on specific, concrete use cases that attract the attention of enterprises, developers, and researchers:

Emphasize that these are non-cherry picked examples

### Secretarial duties

Electronic document information extraction

Transcription

Filling out electronic forms

Converting file formats via Word online

### SysOps

Copying files

Changing system settings

Installing application

### Creative work

Using Blender to generate a scene with overall structure

Writing a story, drawing images and publishing to internet

### Engineering

Using OnShape to make a part that conforms to designs

## Discussion

General discussion

Experimental analyses

### Broader impact

Discuss how revolutionary this project has the potential of becoming.

Enumerate some positive impacts of this technology:

- You can program in your native language!
- Computatrum bridges the divide between actions that are accessible via API and those that require laborious human effort
- TODO
- Has access to the information superhighway (i.e.: the Internet) which gives it alot of world knowledge and modalities

We shouldn't ignore potential negative uses of this technology.

- Captchas. Demonstrate solving one programmatically.
- Increased attack surface for social engineering
- Weapon for hacking, social engineering, and disinformation.
- Malicious content generation. Demonstrate

We reserve AI safety for its own subsection

### Safety

AI Safety has been defined as "TODO"

List safety problems

- perpetuating/magnifying social biases
- unrestricted internet/world access
- replacing jobs

The answers are not clear and we will need to proceed with caution. Justify why I am open-sourcing everything. Maybe cite my paper(s) from CSE-4314.

General discussion on how larger systems will be supervised while given access to the Internet.

Encourage employers to not be quick to replace human processes with machine ones. See AI as a complement rather than replacement.

### Future directions

Fine-tune the vision-language model on computer images (rather than natural images) like email programs, internet sites, etc.

General expectations: More experiments, more complexity, more data.

MAN

Various ideas I want to experiment with

Transfer learning

Full Artificial Experience environment-pipeline

### Acknowledgements

This work evolves out of a long line of research on computer science and artificial intelligence and certainly owes its credit to the millions of underrepresented scientists through history. I also express personal appreciation to neuroscientists Gyorgy Buzsaki and John Beggs, enthusiasts Siraj Raval and Yannic Kilchar, the authors of (Brown et al. 2020), (Goodfellow et al. 2014), and (Hinton 2021), and advisor Deokgun Park, parents Vincent and Tara Valdez, and brother.

### Conclusion

Clearly restate what was accomplished. How it was accomplished. What its impact is. Invite readers to start using the computatrum and contribute to its development.

## Appendix

Appreciation

Footnotes.

Make sure citations are listed

Make sure the discussion is enabled
