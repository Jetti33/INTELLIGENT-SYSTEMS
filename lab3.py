

class RuleBasedSystem:
    def __init__(self):
        self.facts = set()
        self.rules = []

    def add_rule(self, antecedent, consequent):
        """Add a rule in the form IF antecedent THEN consequent."""
        self.rules.append((antecedent, consequent))

    def add_fact(self, fact):
        """Add a fact to the knowledge base."""
        self.facts.add(fact)

    def forward_chain(self):
        """Perform forward chaining until no new facts are inferred."""
        new_fact_found = True
        while new_fact_found:
            new_fact_found = False
            for ante, cons in self.rules:

                if isinstance(ante, list):
                    condition_met = all(condition in self.facts for condition in ante)
                else:
                    condition_met = ante in self.facts

                if condition_met and cons not in self.facts:
                    self.facts.add(cons)
                    new_fact_found = True
                    print(f"Inferred new fact: {cons}")

        print("\nInference complete. Final facts:")
        for fact in sorted(self.facts):
            print(f"- {fact}")



if __name__ == "__main__":
    system = RuleBasedSystem()

    system.add_rule("has_fur", "is_mammal")
    system.add_rule("has_feathers", "is_bird")
    system.add_rule(["is_mammal", "eats_meat"], "is_carnivore")
    system.add_rule(["is_mammal", "has_hooves"], "is_ungulate")
    system.add_rule(["is_carnivore", "has_tawny_color", "has_dark_spots"], "is_cheetah")
    system.add_rule(["is_carnivore", "has_tawny_color", "has_black_stripes"], "is_tiger")
    system.add_rule(["is_ungulate", "has_long_neck", "has_long_legs"], "is_giraffe")
    system.add_rule(["is_ungulate", "has_black_stripes"], "is_zebra")
    system.add_rule("is_bird", "is_animal")
    system.add_rule("is_mammal", "is_animal")
    

    system.add_fact("has_fur")
    system.add_fact("eats_meat")
    system.add_fact("has_tawny_color")
    system.add_fact("has_black_stripes")

    system.forward_chain()
