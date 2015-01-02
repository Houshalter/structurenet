local SWL, parent = torch.class('nn.SWL', 'nn.Module')

function SWL:__init(inputSize, outputSize)
   parent.__init(self)

   --self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   --self.gradWeight = torch.Tensor(outputSize, inputSize):zero()
   self.gradBias = torch.Tensor(outputSize):zero()
   
   self:reset()
end

function SWL:reset(stdv)

end

function SWL:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)

      self.output:resize(nframe, nunit)
      if nunit == 1 then
         -- Special case to fix output size of 1 bug:
         self.output:zero():add(self.bias[1])
         self.output:select(2,1):addmv(1, input, self.weight:select(1,1))
      else
         self.output:zero():addr(1, input.new(nframe):fill(1), self.bias)
         self.output:addmm(1, input, self.weight:t())
      end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function SWL:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end

      return self.gradInput
   end
end

function SWL:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)      
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)
      
      if nunit == 1 then
         -- Special case to fix output size of 1 bug:
         self.gradWeight:select(1,1):addmv(scale, input:t(), gradOutput:select(2,1))
         self.gradBias:addmv(scale, gradOutput:t(), input.new(nframe):fill(1))
      else
         self.gradWeight:addmm(scale, gradOutput:t(), input)
         self.gradBias:addmv(scale, gradOutput:t(), input.new(nframe):fill(1))
      end
   end

end

-- we do not need to accumulate parameters when sharing
SWL.sharedAccUpdateGradParameters = Linear.accUpdateGradParameters