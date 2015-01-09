test1 = {}
dofile("sparseWeightsLinear.lua")
tester = torch.Tester()

--test gradbias
--test gradweights
--test forward prop
--test grad inputs

function test1.TestA()
	local mlp = nn.sparseWeights(5,5)
	mlp.bias = torch.Tensor({1,2,3,4,5})
	mlp.outputWeights = {{1}, {2}, {3.4}, {2.2}, {5,6,7}}
	weightIndex =       {{1}, {2}, {3}, {2}, {5,4,3}}
	mlp.forward(torch.Tensor({2,3,4,5,6}))
	assertTensorEq(mlp.output, torch.Tensor({3,3*2+2,4*3.4+3,4+3*2.2,5+6*5+6*5+4*7}), 0.0000001, "sparseWeight Output incorrect")
	mlp.backward(torch.Tensor({2,3,4,5,6}), torch.Tensor({5,4,3,2,1}))
	assertTensorEq(mlp.gradInput, torch.Tensor({5,2*2.2+4*2,7+3*3.4,6,5}), 0.0000001, "sparseWeight gradInput incorrect")
	assertTableEq(mlp.gradWeight, {{2*5}, {4*3}, {4*3}, {3*2}, {6,5,4}}, 0.0000001, "sparseWeight gradWeight incorrect")
	assertTensorEq(mlp.gradBias, torch.Tensor({5,4,3,2,1}), 0.0000001, "sparseWeight gradBias incorrect")
end

tester:add(test1)
tester:run()