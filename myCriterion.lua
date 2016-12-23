local THNN = require 'nn.THNN'
local myCriterion, parent = torch.class('nn.myCriterion', 'nn.MyCriterion')

function myCriterion:__init(sizeAverage)
	parent.__init(self)
	if sizeAverage ~= nil then
		self.sizeAverage = sizeAverage
	else
		self.sizeAverage = true
	end
	
	self.target = torch.zeros(1):long()
end

function myCriterion:updateOutput(input, target, rewards)
	if type(target) == 'number' then
		if input:type() ~= 'torch.CudaTensor' then
			self.target = self.target:long()
		end
		self.target[1] = target
	elseif target:type() == 'torch.CudaTensor' then
		self.target = target
	else
		self.target = target:long()
	end

	self.rewards = rewards --[batch_size, vocab_size]
	local n_dims = input:dim()
	if target:dim() > 1 then
		assert("multi-target not supported")
	end
	
	if n_dims > 2 then
		assert("input tensor should be 1D or 2D")
	end
	
	self.output = 0.0

	local batch_size = input:size(1)
	local vocab_size = input:size(2)

	for i = 1, batch_size do
		--self.output = self.output - torch.dot(torch.log(input[i]:double()+opt.eps*1e-4), self.rewards[i])
		self.output = self.output - torch.dot(input[i]:double(), self.rewards[i])
	end
	
	if self.sizeAverage then
		self.output = self.output / batch_size
	end
	--[[
	input.THNN.myCriterion_updateOutput(
		input:cdata(),
		self.target:cdata(),
		self.rewards:cdata(),
		self.output_tensor:cdata(),
		self.sizeAverage
	)
	self.output = self.output_tensor[1]
	]]--
	return self.output
end

function myCriterion:updateGradInput(input, target, rewards)
	if type(target) == 'number' then
		if input:type() ~= 'torch.CudaTensor' then
			self.target = self.target:long()
		end
		self.target[1] = target
	elseif target:type() == 'torch.CudaTensor' then
		self.target = target
	else
		self.target = target:long()
	end

	self.rewards = rewards --[batch_size, vocab_size]
	self.gradInput:resizeAs(input:double()):zero()
	local n_dims = input:dim()
	
	if not self.gradInput:isContiguous() then	
		assert("gradInput must be contiguous")
	end
	
	if self.target:dim() > 1 then
		assert("multi-target not supported")
	end

	if n_dims > 2 then
		assert("input tensor should be 1D or 2D")
	end

	local batch_size = input:size(1)
	local vocab_size = input:size(2)
	assert(self.target:size(1) == batch_size)

	for i = 1, batch_size do
		local cp_rewards = torch.Tensor(opt.vocab_size):copy(self.rewards[i])
		if self.sizeAverage then
			self.gradInput[i] = -cp_rewards:cdiv(input[i]:double()+opt.eps*1e-4) / batch_size
			--self.gradInput[i] = -cp_rewards / batch_size
		else
			--self.gradInput[i] = -cp_rewards
			self.gradInput[i] = -cp_rewards:cdiv(input[i]:double()+opt.eps*1e-4)
		end
	end

	--print(self.gradInput)
	
	return self.gradInput
end
	
