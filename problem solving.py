#!/usr/bin/env python
# coding: utf-8

# # Question) What is the difference between list and tuple in Python?
Lists are mutable i.e In list,we can add,modify,remove the data WHEREAS Tuples are immutable i.e we cannot change/update anything in the tuple once we create it.

Syntax: List =[1,'a']  WHEREAS tuple = (1,'a')

Lists have more built-in methods WHEREAS tuple have few methods
# # Question) Explain the concept of PEP 8.
PEP 8 Called as Python Enhancement Proposal 8

By using PEP 8,Python code is written in a way that is easy to understand for both the original author and other developers who might read it.

It improves readibilty.

Examples -
Maximum Line Length - It will not exceed maximum line length
Naming Conventions - Variables, functions, and methods should have lowercase names, with words separated by underscores 
Constants should be in all uppercase with underscores separating words.
Classes should be named using CamelCase.
# # Question) What is the purpose of the _init_ method in Python classes?

# In[1]:


# __init__ method in Python is used for initializing instances of a class. 
# It is called automatically when a new object is created from a class.
# It is used to initialization for the object.

class Myname:
    def __init__(self, name, age):
        # The __init__ method is called when a new Person object is created.
        self.name = name
        self.age = age
person1 = Myname("Akanksha", 22)

print(person1.name)  
print(person1.age)   


# # Question) How does inheritance work in Python? Provide an example.

# In[2]:


# OOPS - Inheritance
# How does inheritance work in Python? Provide an example.
# It allows subclass or derived class to inherit attributes and methods from an existing class (base class or parent class). 
# we can create a subclass by inheriting from a base class, and the subclass can extend or override the functionality of the base class.

class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"



animal = Animal("Generic Animal")
dog = Dog("Buddy")
cat = Cat("Whiskers")


print(dog.speak())
print(cat.speak())


# # Question) Explain the difference between staticmethod and classmethod.

# In[3]:


# A staticmethod is a method which is only to a class rather than an instance of the class.
# It does not have access to the instance or the class itself. It behaves like a regular function.
# We will define a static method using the @staticmethod decorator
# Static methods are often used when the method does not depend on the state of the instance.

# A classmethod is a method which takes the class itself as its first parameter.
# It is used when you want the method to be able to create and return an instance of the class.
# we define a class method using the @classmethod decorator.
# Class methods are called on the class rather than on an instance of the class.

class MyClass:
    class_variable = "I am a class variable"

    def __init__(self, instance_variable):
        self.instance_variable = instance_variable

    @staticmethod
    def static_method(x, y):
        return x + y

    @classmethod
    def create_instance(cls, value):
        return cls(value)

# Uses the static method
result_static = MyClass.static_method(3, 5)
print(f"Result from static method: {result_static}")

# Uses the class method to create an instance
obj = MyClass.create_instance("Hello")
print(f"Instance variable: {obj.instance_variable}")
print(f"Class variable: {obj.class_variable}")


# # Question) What is Polymorphism in Python? Give an example.

# In[5]:


# Polymorphism allows objects of different classes to be treated as objects of a common base class.
# It enables a single interface to represent different types of objects and allows methods to be written that can work with objects of any of those types.
# Polymorphism helps achieve flexibility and code reusability.

# Compile-time Polymorphism - 

# This is achieved through method overloading and operator overloading.
# Method overloading occurs when multiple methods in the same class have the same name but different parameter lists 


# Run-time Polymorphism (Dynamic Binding):

# This is achieved through method overriding.
# Method overriding occurs when a method in a subclass has the same name and parameter list as a method in its superclass.
# The method in the subclass overrides the method in the superclass.

class Animal:
    def speak(self):
        pass  # Abstract method, to be overridden by subclasses

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

class Duck(Animal):
    def speak(self):
        return "Quack!"

# Function that demonstrates polymorphism
def animal_says(animal):
    return animal.speak()

# Creating instances of different subclasses
dog_instance = Dog()
cat_instance = Cat()
duck_instance = Duck()

# Calling the function with different instances
print(animal_says(dog_instance)) 
print(animal_says(cat_instance))  
print(animal_says(duck_instance)) 


# # Question) How do you handle exceptions in Python?
Handling exceptions in Python involves using a combination of the try, except, else, and finally blocks. 

Try Block:

The try block contains the code where you anticipate an exception might occur.
If an exception occurs in the try block, the code inside the except block is executed.

Except Block:

The except block catches and handles the specific exceptions raised in the try block.
You can have multiple except blocks to handle different types of exceptions.

Else Block (Optional):

The else block, if present, is executed if no exceptions occur in the try block.
It is often used for code that should run only when no exceptions occur.

Finally Block (Optional):

The finally block, if present, is always executed whether an exception occurs or not.
It is often used for cleanup code that should run regardless of whether an exception occurred.
# # Question) Explain the Global Interpreter Lock (GIL) in Python.
The Global Interpreter Lock (GIL) is a mechanism used in the implementation of the CPython interpreter (the most widely used implementation of Python) to synchronize access to Python objects, preventing multiple native threads from executing Python bytecodes at once. The GIL is specific to CPython and is not present in all Python implementations.

Single Thread Execution
Impact on I/O-Bound Tasks
Impact on Multithreading
Use of Multiprocessing
# # Question) What is a decorator in Python? Provide an example.

# In[6]:


# A decorator in Python is a design pattern that allows you to extend or modify the behavior of a callable (functions or methods) without modifying its actual code.
# Decorators are applied using the @decorator syntax or by using the decorator function explicitly
# Decorators are often used for aspects such as logging, timing, memoization, and more.

# Decorator function
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

# Applying the decorator using the @ syntax
@my_decorator
def say_hello():
    print("Hello!")

# Calling the decorated function
say_hello()


# # Question) How do you implement encapsulation in Python?
# 

# In[9]:


class BankAccount:
    def __init__(self, account_holder, balance=0):
        self.__account_holder = account_holder  # Private attribute
        self.__balance = balance  # Private attribute

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"Deposited ${amount}. New balance: ${self.__balance}")
        else:
            print("Invalid deposit amount.")

    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            print(f"Withdrew ${amount}. New balance: ${self.__balance}")
        else:
            print("Invalid withdrawal amount.")

    def get_balance(self):
        return self.__balance

    def get_account_holder(self):  # Corrected method name
        return self.__account_holder

# Creating an instance of the BankAccount class
account1 = BankAccount(account_holder="John Doe", balance=1000)

# Accessing attributes through methods (encapsulation)
print(f"Account holder: {account1.get_account_holder()}")
print(f"Current balance: ${account1.get_balance()}")

# Depositing and withdrawing money
account1.deposit(500)
account1.withdraw(200)

# Trying to withdraw more money than the balance
account1.withdraw(10000)


# # Question) Explain the concept of duck typing
Duck typing is a programming concept used in dynamic languages, such as Python, where the type or class of an object is determined by its behavior (methods and properties) rather than its explicit inheritance or type declaration.

duck typing allows you to write flexible and generic code that can work with different types of objects as long as they support the required methods or attributes. This promotes code reusability and flexibility.
# In[10]:


class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

class Duck:
    def speak(self):
        return "Quack!"

def make_sound(animal):
    return animal.speak()

dog_instance = Dog()
cat_instance = Cat()
duck_instance = Duck()

print(make_sound(dog_instance))  
print(make_sound(cat_instance))  
print(make_sound(duck_instance))  


# # Question) What is the difference between append() and extend() methods for lists?

# In[12]:


# #append():

# The append() method is used to add a single element to the end of a list.
# It appends it to the end of the list.
# If the argument is another list, it will be added as a single element at the end of the original list.

# extend():

# The extend() method is used to add elements from an iterable (e.g., another list, tuple, string) to the end of a list.
# It takes an iterable as its argument and adds each element from the iterable to the end of the list.
# It modifies the original list in-place.

# Using append() to add a single element to the end of a list
numbers_list = [1, 2, 3]
numbers_list.append(4)
print("After append:", numbers_list)  # Output: [1, 2, 3, 4]

# Using extend() to add elements from another iterable (list) to the end of the list
more_numbers = [5, 6, 7]
numbers_list.extend(more_numbers)
print("After extend:", numbers_list)  # Output: [1, 2, 3, 4, 5, 6, 7]


# # Question) How does the with statement work in Python?
The with statement in Python is used to simplify the management of resources, such as files or network connections, by ensuring that certain operations are performed before and after a block of code. It is commonly used for working with external resources that need to be acquired and released properly.
The primary purpose of the with statement is to provide a clean and concise syntax for the common try-finally pattern, where you ensure that certain cleanup or finalization steps are taken even if an exception occurs.
# In[14]:


class MyContextManager:
    def __enter__(self):
        print("Entering the context")
        return self  # The object to be used in the with block

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting the context")

# Using the custom context manager
with MyContextManager() as context:
    print("Inside the with block")


# # Question) Discuss the use of self in Python classes.
In Python, self is a convention used as the first parameter of instance methods in a class. 
It refers to the instance of the class itself and allows you to access and modify attributes of that instance. 
The use of self is a fundamental concept in Python's object-oriented programming paradigm.

When you define a class and create an instance of that class, the instance is passed as the first parameter to every method defined within the class. By convention, this parameter is named self, but you could technically name it anything (though it's strongly recommended to stick with self for clarity and readability).
# # Question) Explain the purpose of the _slots_ attribute.
The __slots__ attribute in Python is used to explicitly declare and restrict the attributes that instances of a class can have. By using __slots__, you define a fixed set of allowed attributes for instances of the class, and any attempt to create or assign a new attribute not listed in __slots__ will result in an AttributeError. This can be useful for memory efficiency and can also prevent accidental creation of new attributes.

# In[15]:


class Person:
    __slots__ = ('name', 'age')

    def __init__(self, name, age):
        self.name = name
        self.age = age

# Creating an instance of the Person class
person = Person(name="Alice", age=30)

# Accessing attributes
print(person.name)  
print(person.age)   


# # Question) What is the difference between an instance variable and a class variable?
The main differences between instance variables and class variables in Python are related to their scope, usage, and how they are accessed.

Scope:

Instance Variable:
Instance variables are specific to each instance of a class.
They are defined inside methods using the self keyword and are unique to each object created from the class.

Class Variable:
Class variables are shared among all instances of a class.
They are defined outside of any method in the class and are associated with the class itself rather than with instances.
Access:

Instance Variable:
Accessed using the instance of the class (self).
Each instance has its own copy of the instance variables.
Modifications to an instance variable affect only that particular instance.
Class Variable:
Accessed using the class name.
Shared among all instances of the class.
Modifications to a class variable affect all instances of the class.
Purpose:

Instance Variable:
Represents the state or attributes of a specific instance.
Varied values for each instance.
Defined inside methods using self.
Class Variable:
Represents attributes or properties shared by all instances.
Common value for all instances.
Defined outside methods at the class level.
# # Question) How do you implement Encapsulation, Abstraction,Polymorphism?

# In[19]:


# Encapsulation
class BankAccount:
    def __init__(self, account_holder, balance=0):
        self.__account_holder = account_holder  # Private attribute
        self.__balance = balance  # Private attribute

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"Deposited ${amount}. New balance: ${self.__balance}")
        else:
            print("Invalid deposit amount.")

    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            print(f"Withdrew ${amount}. New balance: ${self.__balance}")
        else:
            print("Invalid withdrawal amount.")

    def get_balance(self):
        return self.__balance

    def get_account_holder(self):  # Corrected method name
        return self.__account_holder

# Creating an instance of the BankAccount class
account1 = BankAccount(account_holder="John Doe", balance=1000)

# Accessing attributes through methods (encapsulation)
print(f"Account holder: {account1.get_account_holder()}")
print(f"Current balance: ${account1.get_balance()}")

# Depositing and withdrawing money
account1.deposit(500)
account1.withdraw(200)


# In[20]:


# Abstraction
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius**2

# Using abstraction to create and use shapes
rectangle = Rectangle(length=5, width=3)
circle = Circle(radius=4)

print(f"Area of the rectangle: {rectangle.area()}")
print(f"Area of the circle: {circle.area()}")


# In[22]:


# Polymorphism
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

# Using polymorphism to treat objects uniformly
def make_sound(animal):
    return animal.speak()

# Creating instances of different classes
dog_instance = Dog()
cat_instance = Cat()

# Using polymorphism to call the 'speak' method
print(make_sound(dog_instance))  
print(make_sound(cat_instance))  


# # Question) How do you Implement single level Inheritance, multiple level inheritance, multi level inheritance, Hybrid Inheritance

# In[24]:


# Single level inheritance
class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):
    def bark(self):
        print("Dog barks")

# Creating an instance of Dog
my_dog = Dog()
my_dog.speak() 
my_dog.bark()  


# In[25]:


# Multiple level inheritance
class Animal:
    def speak(self):
        print("Animal speaks")

class Mammal(Animal):
    def run(self):
        print("Mammal runs")

class Dog(Mammal):
    def bark(self):
        print("Dog barks")

# Creating an instance of Dog
my_dog = Dog()
my_dog.speak()  # Inherited from Animal
my_dog.run()    # Inherited from Mammal
my_dog.bark()   # Specific to Dog


# In[26]:


# Multi level inheritance
class Animal:
    def speak(self):
        print("Animal speaks")

class Mammal(Animal):
    def run(self):
        print("Mammal runs")

class Dog(Mammal):
    def bark(self):
        print("Dog barks")

class Bulldog(Dog):
    def guard(self):
        print("Bulldog guards")

# Creating an instance of Bulldog
my_bulldog = Bulldog()
my_bulldog.speak()  # Inherited from Animal
my_bulldog.run()    # Inherited from Mammal
my_bulldog.bark()   # Inherited from Dog
my_bulldog.guard()  # Specific to Bulldog


# In[27]:


# Hybrid Inheritance
class Animal:
    def speak(self):
        print("Animal speaks")

class Mammal(Animal):
    def run(self):
        print("Mammal runs")

class Bird(Animal):
    def fly(self):
        print("Bird flies")

class Bat(Mammal, Bird):
    def navigate(self):
        print("Bat navigates in the dark")

# Creating an instance of Bat
my_bat = Bat()
my_bat.speak()     # Inherited from Animal
my_bat.run()       # Inherited from Mammal
my_bat.fly()       # Inherited from Bird
my_bat.navigate()  # Specific to Bat


# In[ ]:




