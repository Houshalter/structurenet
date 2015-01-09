local sparseWeights, Parent = torch.class('nn.sparseWeights', 'nn.Module')

function sparseWeights:updateOutput(input)
	self.output:zero()
	for outnum,weights in ipairs(self.outputWeights) do
		local total = 0
		for weightnum, weight in ipairs(weights) do
			total = total + input[self.weightIndex[outnum][weightnum]]*weight
		end
		self.output[outnum] = total
	end
	self.output:add(self.bias)
end

function sparseWeights:updateGrad(input, gradOutput)
	self.gradInput:zero()
	for outnum,grad in ipairs(gradOutput) do
		if not self.gradWeight[outnum] then self.gradWeight[outnum] = {} end
		for weightnum,weight in ipairs(self.outputWeights[outnum]) do
			if not self.gradWeight[outnum][weightnum] then self.gradWeight[outnum][weightnum] = 0 end
			local index = self.weightIndex[outnum][weightnum]
			self.gradInput[index] = weight*grad
			self.gradWeight[outnum][weightnum] = input[index]*grad+self.gradWeight[outnum][weightnum]
		end
	end
	self.gradBias:add(gradOutput)
end

function sparseWeights:backward(input, gradOutput)
	updateGrad(input, gradOutput)
	return self.gradInput
end

function sparseWeights:__init(sizeInput, sizeOutput)
	Parent.__init(self)
	self.gradInput = torch.Tensor(sizeInput)
	self.gradWeight = {}
	self.output = torch.Tensor(sizeOutput)
	self.bias = torch.Tensor(sizeOutput):randn()
end

--strNet = nn.Sequential()
--str:add(nn.sparseWeights())