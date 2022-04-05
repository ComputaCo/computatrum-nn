from __future__ import annotations
from copy import deepcopy
import random
from typing_extensions import Self, Protocol
from box import Box
from typing import Callable, Dict, Iterator, List, Optional, Set, SupportsInt, Tuple, Union, runtime_checkable

"""
Reserved scope variables (used for Node parameters):
- template
- N
- scope
- children
- updates
- items
- named_items
"""


def leval(var, **kwargs):
    """Lazy eval.

    Gets the value behind a variable `var` that may be
    - a zero position arg function
    - a string that identifies a variable in the local or global scope
    - a raw string

    NOTE: the zero position arg function is a special case. `leval` will NOT: 
    - evaluate a statement expressed as a string ("print('hello')")
    - evaluate a zero position arg function expressed as a string ("exit")

    Args:
        var (str): Variable to evaluate.
        **kwargs: Keyword arguments to pass to the variable if it is a function.

    Returns:
        The evaluated value of the function 
        or the local or global variable
        or the raw string.

    Raises:
        Exception: If the variable is not found in the local or global scope.
    """
    if callable(var):
        return var(**kwargs)
    elif isinstance(var, str):
        l, g = locals(), globals()
        if var in l:
            return l[var]
        elif var in g:
            return g[var]
        else:
            return var
    else:
        raise ValueError(f'Invalid lazy variable: {var}')


def sel(*options, **kwargs):
    """Selects a random option from a list of options.

    Args:
        options (List[str]): List of options.
        **kwargs: Keyword arguments to invoke the option with, if applicable.

    Returns:
        str: Randomly selected option.

    Raises:
        Exception: If no valid options are found.
    """
    # sample until we get a valid option or run out of options
    while True:
        try:
            return leval(random.choice(options), **kwargs)
        except IndexError:
            raise Exception('No valid options found.')


class Node:

    global_scope: Box = Box({})

    scope_key: str = "Node"

    children: List[Node]
    scope: Box

    def __init__(self,
                 children: List[Node],
                 scope: Box = None,
                 updates: dict = {}) -> None:
        """Initializes a node.

          Args:
              children (List[Node]): List of children.
              scope (Box, optional): Variable scope. Defaults to global scope.
              updates (dict, optional): Updates to shared scope. Defaults to {}.
        """
        self.children = children
        if scope is None:
            scope = Node.global_scope
        # make updates before copying and freezing scope
        # so they get passed on to siblings and parents
        self.scope.update(updates)
        self.scope = deepcopy(scope).freeze()

    def render(self) -> str:
        """Renders the node to a string.
        In most cases, render should be performed using post-order traversal.
        Nodes may use information in their scope to render themselves.

        Returns:
            str: Rendered string.
        """
        raise Exception('Not implemented.')

    def execute(self, env, **kwargs) -> Optional[Dict]:
        """Execute the node.

        Args:
            env: Environment to execute on.
            **kwargs: Additional keyword arguments to pass to children.
        """
        for c in self.children:
            scope = c.execute(scope=scope, env=None) or scope
        return scope

    def __repr__(self) -> str:
        return self.render()

    def __str__(self) -> str:
        return self.render()


Template = Union[Set['Template'], Tuple['Template'], Dict[str, 'Template'],
                 Iterator['Template'], List['Template'], Node, str]


class StringNode(Node):

    def __init__(self,
                 string: str,
                 scope: Box = None,
                 updates: dict = {}) -> None:
        """Initializes a string node.

        Args:
            string (str): String to render.
            scope (Box, optional): Variable scope. Defaults to global scope.
            updates (dict, optional): Updates to shared scope. Defaults to {}.
        """
        super().__init__([], scope, updates)
        self.string = string

    def render(self) -> str:
        return self.string


class EmptyNode(StringNode):

    def __init__(self,
                 scope: Box = None,
                 updates: dict = {}) -> None:
        """Initializes an empty node.

        EmptyNode are useful for
        - building optional nodes
        - setting arbitrary scope values

        Args:
            scope (Box, optional): Variable scope. Defaults to global scope.
            updates (dict, optional): Updates to shared scope. Defaults to {}.
        """
        super().__init__('', scope, updates)


@runtime_checkable
class LazyTemplateFn(Protocol):
    def __call__(self, scope: Box, **kwargs) -> Node: pass


class TemplateNode(Node):
    """Template Node
    Convenience constructor for psuedo context-free grammars

    TemplateNode (T) is an 'abstract' class for the following subclasses:
    - ConcatNode (C)
      - RepeatNode (R)
      - UnionNode (U)
        - OptionalNode (O)
    The symbol in parenthesis denotes a shorthand for the corresponding type.
    TemplateNode is 'abstract' in the sense that it doesn't even have a
    constructor. The __new__ method is overridden to return a subclass of
    TemplateNode depending on the type of the template.

    The `template` arg can be a nested structure of any of the following:
    - node: Node: A single node.
    - v: str: If the string evaluates to a `Template`, then that object, otherwise,
        a StringNode with the string value.
    - fn: LazyTemplateFn: lazy template generation. kwargs are inherited from the
        parent TemplateNode. Useful for recursive grammars. Evaluated on initialization.
    - {*nodes: List[Node]}: Union of nodes. Chooses one at random. Converted to
        a UnionNode on initialization.
    - (*nodes: List[Node]|int): concatenates nodes. 0 or 1 arguments may be an int.
        If an int is passed, then that many nodes are selected at random to become 
        children -- effectively making a semi-optional node. Converted to ConcatNode
        on initialization.
    - {N: int, **key: List[Node]}: concatenates nodes and assigns unique names to children. 
        0 or 1 arguments may be an int. If N is supplied, then N nodes are selected
        at random at become children -- effectively making a semi-optional node.
        Converted to ConcatNode on initialization.
    - [*nodes: List[Node]]: Optional concatenation of nodes. If no nodes are passed,
        then the node is converted to an EmptyNode on initialization. Otherwise, it is
        converted to an OptionalNode containing ConcatNode on initialization.
    - (|*nodes: List[Node]|) | Iter[Node]: Lazy concatenation of nodes. Values inside
        banana brackets (only supported in the coconut language) are not evaluated until 
        the node is initialized. This allows defining recursive grammars. Converted to a
        ConcatNode on initialization.

    Except for dict structures, keys for children are determined by the child's `scope_key` attribute.
    The scope is updated by: `scope[node.scope_key] = node`. By default, the scope_key is the class name.
    """

    def __new__(cls: type[Self], template: Template, **kwargs) -> Self:
        """Constructor for TemplateNode.

            Returns:
                A subclass of TemplateNode: New TemplateNode.
        """
        if isinstance(template, Node):
            return Node  # cls.__new__(cls, [template], **kwargs)
        elif isinstance(template, str):
            return leval(template, **kwargs)
        elif issubclass(template, LazyTemplateFn):
            return template(**kwargs)
        elif isinstance(template, set):
            return UnionNode(list(template), **kwargs)
        elif isinstance(template, tuple):
            ints = filter(lambda x: isinstance(x, int), template)
            N = ints[0] if any(ints) else None
            return ConcatNode(list(template), N=N, **kwargs)
        elif isinstance(template, dict):
            kwargs.update(template)
            return ConcatNode(**kwargs)
        elif isinstance(template, list):
            return OptionalNode(tuple(template), **kwargs)
        elif isinstance(template, Iterator):
            return TemplateNode(tuple(template), **kwargs)
        else:
            raise Exception('Invalid template type: {}'.format(type(template)))


class ConcatNode(TemplateNode):

    def __init__(self,
                 *items: List[Template],
                 N: int = None,
                 scope: Box = None,
                 updates: dict = {},
                 **named_items: Dict[str, Node]) -> None:
        """Initializes a ConcatNode.

        A ConcatNode is a node that concatenates a list of nodes in the render method.

        Args:
            *items (List[Template]): List of items. Optional if supplying named items.
            N (int, optional): Number of items to choose from. Defaults to None (all items).
            scope (Box, optional): Variable scope. Defaults to global scope.
            updates (dict, optional): Updates to shared scope. Defaults to {}.
            **named_items (Dict[str, Node]): Children with named keys. 
        """
        assert not (named_items and items), \
            "Cannot provide both named and positional children"

        if len(items) == 1 and isinstance(items[0], dict):
            named_items = items[0]

        if items:
            children = map(TemplateNode, items)
        elif named_items:
            children = named_items.values()
            children = map(TemplateNode, children)
            for name, child in zip(named_items.keys(), children):
                child.scope_key = name
        else:
            children = [EmptyNode()]

        if N is not None:
            assert N > 0, "N must be greater than 0"
            assert len(children) > N, \
                "N must be less than the number of children"
            # sample children without altering relative order
            indeces = random.sample(range(len(children)), N)
            indeces.sort()
            children = [children[i] for i in indeces]

        super().__init__(
            children=children,
            scope=scope,
            updates=updates)

    def render(self) -> str:
        return ''.join(map(lambda x: x.render(), self.children))


class RepeatNode(ConcatNode):

    def __init__(self,
                 item: Template,
                 sep: Template = None,
                 last_sep: Template = None,
                 *,
                 N: int = None,
                 exp_lambda: float = 0.333,
                 min_count: int = 0,
                 max_count: int = None,
                 scope: Box = None,
                 updates: dict = {}) -> None:
        """Initializes a RepeatNode.

        A RepeatNode is a node that makes 1 or more repitions of a node in the render
        method. Note that the repitition happens at initialization time, so the repeated
        nodes will be different objects.

        Under the hood, initalizes a ConcatNode with children given as a 
        repitition of `item` separated by `sep` (if given) and the last item
        separated by `last_sep` (if given). 

        If N is given, then the item is repeated exactly N times. 
        Otherwise, the item is repeated with repititions sampled from an
        Exponential distribution bound within [min_count, max_count].

        The mean of the exponential distribution is 1 / exp_lambda.

        Args:
            item (Template): Item to repeat.
            sep (Template, optional): Separator between items. Defaults to None.
            last_sep (Template, optional): Separator between last and second to last items.
                Defaults to None.
            N (int, optional): Number of repetitions. Defaults to None. If specified,
                then poison parameters, min_count, and max_count are ignored.
            exp_lambda (float, optional): Lambda for Exponential distribution. Defaults to 1.0.
                Lower values mean more frequent repetitions. Ignored if N is given.
            min_count (int, optional): Minimum number of repetitions. Defaults to 0.
            max_count (int, optional): Maximum number of repetitions. Defaults to None.
            scope (Box, optional): Variable scope. Defaults to global scope.
            updates (dict, optional): Updates to shared scope. Defaults to {}.
        """

        if N is None:
            N = int(random.expovariate(exp_lambda))
            N = max(N, min_count)
            if max_count is not None:
                N = min(N, max_count)

        items = []
        for i in range(N):
            items.append(item)
            if i < N - 2 and sep is not None:
                items.append(sep)
            elif i == N - 2:
                if last_sep is not None:
                    items.append(last_sep)
                elif sep is not None:
                    items.append(sep)

        super().__init__(*items, scope=scope, updates=updates)


class UnionNode(ConcatNode):

    def __init__(self,
                 *items: List[Template],
                 scope: Box = None,
                 updates: dict = {},
                 **named_items: Dict[str, Node]) -> None:
        """Initializes a UnionNode.

        A UnionNode is a node that chooses one of its children in the render method.
        Note that the choice happens at initialization time, so the rendner is deterministic.

        Under the hood, initializes a ConcatNode with N=1 (i.e.: only 1 item selected).

        Args:
            *items (List[Template]): List of items. Optional if supplying named items.
            scope (Box, optional): Variable scope. Defaults to global scope.
            updates (dict, optional): Updates to shared scope. Defaults to {}.
            **named_items (Dict[str, Node]): Children with named keys. 
        """
        super().__init__(
            *items,
            N=1,
            scope=scope,
            updates=updates,
            **named_items)

    def render(self) -> str:
        return random.choice(self.children).render()


class OptionalNode(UnionNode):

    def __init__(self,
                 item: Template,
                 scope: Box = None,
                 updates: dict = {}) -> None:
        """Initializes a OptionalNode.

        A OptionalNode is a node that represents the union of a given node and an empty node.
        Note that the choice happens at initialization time, so the rendner is deterministic.

        Under the hood, initializes a UnionNode with an EmptyNode or the item

        Args:
            item (Template): Item to repeat.
            scope (Box, optional): Variable scope. Defaults to global scope.
            updates (dict, optional): Updates to shared scope. Defaults to {}.
        """
        super().__init__(
            item,
            EmptyNode(),
            scope=scope,
            updates=updates)


# convenience types
N = Node
S = StringNode
E = EmptyNode
T = TemplateNode
C = ConcatNode
R = RepeatNode
U = UnionNode
O = OptionalNode


# this is what an example syntax definition should look like:
DOWN = U("down", "press")
UP = {"up", "release"}

BUTTON_VB = U("click", DOWN, UP)
BUTTON_SIDE = {"left", "right", "middle"}

BUTTON_ACTION = {
    (BUTTON_VB, "the", BUTTON_SIDE, "button"),
    ([BUTTON_SIDE], BUTTON_VB)
}

ACTION = {
    {"mouse": BUTTON_ACTION},
    {"keyboard": 'press A'},
}

TASK = R(ACTION, sep=", ", last_sep=" and ", exp_lambda=0.1)

"""TODO: refactor much of init logic into separate generate() function

# 1. define syntax
# 2. generate templates
# 3. render and execute templates

tasks = []
for i in range(10):
    task = TASK.generate(scope=Box(), updates={}, **kwargs)  
    # scope is saved on each node, so we just make an anonymous one
    # updates for task-level parametrizations
    # **kwargs for dev-level customization (like passing a param to a deep node)
    tasks.append(task)

for task in tasks:
    print(task.render())
    task.execute(env=env)

'''Init vs. Generate
- shared scope reference is passed around during init
- scope is edited, copied, and frozen during generate
- scope is read during render and execute
'''
"""
