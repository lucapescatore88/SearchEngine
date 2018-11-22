from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("name",help="The name of the person")
parser.add_argument("surname",help="The surname of the person")
parser.add_argument("--midname",default="",help="The mid of the person")
parser.add_argument("--nationality",default="",help="Nationality")
parser.add_argument("--domicile",default="",help="Domicile")