function updateOutput(input)
	self.output:zero()
	for outnum,weights in ipairs(self.outputWeights) do
		local total = 0
		for weightnum, weight in ipairs(weights) do
			total = total + input[self.weightIndex[outnum][weightnum]]*weight
		end
		self.output[outnum] = total
	end
end

function updateGradInput(input, gradOutput)
	self.gradInput:zero()
	for outnum,grad in ipairs(gradOutput) do
		for weightnum,weight in ipairs(self.outputWeights[outnum]) do
			self.gradInput[self.weightIndex[outnum][weightnum]]
		end
	end
end

function accGradParameters(input, gradOutput)
	
end