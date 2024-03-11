class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):
    # The child's __init__() function overrides the 
    # inheritance of the parent's __init__() function.
    def __init__(self, name):
        super().__init__(name)
        self.race = 'dog'

    def talk(self):   
        # specialize the behaviour of this subclass
        print('Woof!')

class Cat(Animal):
    def __init__(self, name):
        super().__init__(name)
        self.race = 'cat'

    def purrr(self):
        print('purrr...')

