



class grid: 
    
    def __init__(self, params_dict): 
        self.params_dict = params_dict

    def get_combinations(self):
        keys, values = zip(*self.params_dict.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def __repr__(self):
        return f"HyperparameterGrid({self.param_dict})"