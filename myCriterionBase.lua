local my_Criterion = torch.class('nn.MyCriterion')

function my_Criterion:__init()
   self.gradInput = torch.Tensor()
   self.output = 0
end

function my_Criterion:updateOutput(input, target, rewards)
end

function my_Criterion:forward(input, target, rewards)
   return self:updateOutput(input, target, rewards)
end

function my_Criterion:backward(input, target, rewards)
   return self:updateGradInput(input, target, rewards)
end

function my_Criterion:updateGradInput(input, target, rewards)
end

function my_Criterion:clone()
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   return clone
end

function my_Criterion:type(type, tensorCache)
   assert(type, 'Criterion: must provide a type to convert to')
   -- find all tensors and convert them
   for key,param in pairs(self) do
      self[key] = nn.utils.recursiveType(param, type, tensorCache)
   end
   return self
end

function my_Criterion:float()
   return self:type('torch.FloatTensor')
end

function my_Criterion:double()
   return self:type('torch.DoubleTensor')
end

function my_Criterion:cuda()
   return self:type('torch.CudaTensor')
end

function my_Criterion:__call__(input, target, rewards)
   self.output = self:forward(input, target, rewards)
   self.gradInput = self:backward(input, target, rewards)
   return self.output, self.gradInput
end
