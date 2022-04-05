# GUI Bot Language Tasks

See <https://aclanthology.org/2021.findings-acl.187.pdf> for more ideas

## Context-free grammar

### Definitions

E = ""
S = " "
HYPHEN = S | -
CMD_TERMINATOR = E|,|.|!
AND = "and"|"&"
OR = "or"|"|"

PARALLEL_ADV = at the same time | simultaneously
INTENSITY_ADV = more | less | not as much
SPEED_ADV = quickly | slowly
CAUTION_ADV = carefully | with caution
MOVEMENT_ADV = CAUTION_ADV | SPEED_ADV | INTENSITY_ADV
ADV = PARALLEL_ADV | INTENSITY_ADV | SPEED_ADV | MISC_ADV

SPEED_VB = faster | slower | quick

REG_KEY := all non-modifier keys
MOD_KEY := all modifier keys
KEY := REG_KEY | MOD_KEY
KEY_COMBO_NO_SEP := MOD_KEY? KEY_COMBO_NO_SEP KEY? | E
KEY_COMBO_PLUS := MOD_KEY + KEY_COMBO_PLUS | KEY_COMBO_PLUS + KEY | KEY | E
KEY_COMBO_AND_PARTIAL := MOD_KEY AND KEY_COMBO_AND_PARTIAL | KEY_COMBO_AND_PARTIAL AND KEY | KEY | E
KEY_COMBO_AND := KEY_COMBO_AND_PARTIAL PARALLEL_PREPOSITION?
KEY_COMBO := KEY_COMBO_NO_SEP | KEY_COMBO_PLUS | KEY_COMBO_AND
KEY_ACTION := press | release

BUTTON_SIDE := left | right | middle
BUTTON_NAME := BUTTON_SIDE (mouse?) button
BUTTON_ABBREV := LMB | RMB | MMB
BUTTON_ACTION := press|release|click|double click|triple click

SCROLL_DIR := up|down

CARDINAL_POS := top | bottom | left side | right side
CARDINAL_DIR := (up | down | left | right)(E | ward | wards)
TEXT_POS := start | beginning | middle | halfway(HYPHEN)point | end
TEXT_DIR := (forward | backward)(s?)
DIR := CARDINAL_DIR | TEXT_DIR

### Motor-level tasks

Examples: LMB down, press A, backspace

MOTOR_ACTION :=
  // keyboard
  KEY_ACTION |
  KEY_COMBO |
  SPEED_ADV? KEY_ACTION KEY_COMBO |
  KEY_ACTION KEY_COMBO SPEED_ADV? |
  SPEED_ADV? KEY_ACTION the KEY_COMBO key(s?) |
  KEY_ACTION the KEY_COMBO key(s?) SPEED_ADV? |
  SPEED_ADV? KEY_ACTION key(s?) KEY_COMBO |
  KEY_ACTION key(s?) KEY_COMBO SPEED_ADV? |
  // mouse buttons
  BUTTON_ACTION SPEED_ADV? |
  SPEED_ADV? BUTTON_ACTION |
  BUTTON_SIDE |
  BUTTON_NAME |
  BUTTON_ABBREV |
  SPEED_ADV? BUTTON_ACTION the (BUTTON_NAME|BUTTON_ABBREV) |
  BUTTON_ACTION the (BUTTON_NAME|BUTTON_ABBREV) SPEED_ADV? |
  (BUTTON_SIDE | BUTTON_NAME | BUTTON_ABBREV) BUTTON_ACTION |
  // scroll wheel
  SPEED_ADV? scroll SCROLL_DIR? |
  scroll SCROLL_DIR? SPEED_ADV? |
  // mouse movement
  DIR |
  move (the mouse)? DIR MOVEMENT_ADV? |
  MOVEMENT_ADV? move (the mouse)? DIR  |
  (move (the mouse)?)? DIR |
  E

### Low-level tasks

Examples: type "hello", click the blue triangle, delete that word

- pages of just different buttons that can be identified by text, color, shape, or image.

### UI understanding

Examples: check the checkbox, scroll the scrollbar, open a window

- traditional forms (input fields and a submit)

### App understanding

Examples: open email, go to "google.com", copy the files in /tmp to /home/jacob/backup

### Language skills

Examples: email Phil that I'll be out tomorrow

### Visual skills

Examples: less general than the others; draw a picture of a dog

### Instruction following

Examples: follow along to the YouTube tutorial, follow these steps to instal tensorflow

## GUI Bot

The most recently specified key / button / direction / action / etc. is usually assumed if one of the variables are missing. Exceptions are "Press A. Press B. Press C. Press. Press. Press" where D, E, and F are implicit, but the bot cannot do this.

Actions are executed by a bot that keeps track of:

- key
- key action
- mouse button
- mouse button action
- scroll direction
- mouse direction
- intensity
- speed
- caution

VP :=
  SPEED_VB |
  MOTOR_ACTION

## Post-processing

1. Random spelling correction
2. Random synonym replacement (use LM to make sure semantic meaning stays similar)
3. Translation to randomly selected language (often based on OS)
4. Random misspellings (I'm not sure if this is a good idea)
