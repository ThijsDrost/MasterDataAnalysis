"""
This subpackage contains classes for representing chemical reactions and surface reactions.
"""

# TODO: is this module usefull?
from General.Data_handling.Reactions import Reaction


class ReactionEquation:
    def __init__(self, reactants, products, equation):
        self.reaction = Reaction(reactants, products)
        self.equation = equation


class ElectronImpactReaction(ReactionEquation):
    def __init__(self, reactants, products, equation, reaction_type):
        super().__init__(reactants, products, equation)
        self.reaction_type = reaction_type
        if reaction_type not in ('Ionization', 'Attachment'):
            raise ValueError(f'Invalid reaction type: {reaction_type}, must be either "Ionization" or "Attachment"')


class HeavySpeciesReaction(ReactionEquation):
    def __init__(self, reactants, products, equation):
        super().__init__(reactants, products, equation)


class SurfaceReaction:
    def __init__(self, reactants, products, sticking_coefficient, emission_coefficient, electron_energy):
        self.reaction = Reaction(reactants, products)
        self.sticking_coefficient = sticking_coefficient
        self.emission_coefficient = emission_coefficient
        self.electron_energy = electron_energy
