class Optimizer(object):
    def __init__(self, update_rule, **kwargs):
        """
        Set self.update
        Set parameters from kwargs(variable keyword parameters) according to update type
        """
        self.update = update_rule

        if self.update == "sgd":
            self.lr = kwargs['lr']

    def step(self, model):
        """
        Returns proper update according to the string in self.update

        Input:
        model: NN model

        Output:
        None
        """
        if self.update == "sgd":
            for key, _ in model.state_dict().items():
                model.__dict__.get(key).value += self.update_sgd(
                    model.__dict__.get(key).grad, self.lr
                )
            model.update_state_dict()
    def update_sgd(self, lr, gradient):
        """
        Update function for sgd.
        Make sure you update the moving averages for 'layer' inside this function.

        Inputs:
        layer: layer for which the update has to be returned
        model: NN model to be updated (you need this to access 'state' and update histories)
        lr: learning rate

        Outputs:
        update (Tensor)
        """

        return -lr * gradient

    def zero_grad(self, model):
        for key, _ in model.state_dict().items():
            model.__dict__.get(key).zero_grad()
