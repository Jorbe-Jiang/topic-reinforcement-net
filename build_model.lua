----------------------------------------------------
--model structure
----------------------------------------------------
require('myCriterionBase')
require('myCriterion')
require('myMaskZeroCriterion')
require('mySequencerCriterion')

local emb = nn.LookupTableMaskZero(opt.vocab_size, opt.word_dim):setMaxNorm(2)
if opt.load_embeddings == 1 then
	print('Loading pre-trained word vectors from: '..opt.load_embeddings_file)
	local trained_word_vecs = npy4th.loadnpy(opt.load_embeddings_file)
	emb.weight = trained_word_vecs
end

function RNN_elem(recurrence)
	local utterance_rnn = nn.Sequential()
	local hred_enc_embeddings = nn.LookupTableMaskZero(opt.vocab_size, opt.word_dim):setMaxNorm(2)
	hred_enc_embeddings.weight = emb.weight:clone()
	utterance_rnn:add(hred_enc_embeddings)

	local rnn = recurrence(opt.word_dim, opt.enc_hidden_size)
	utterance_rnn:add(nn.Sequencer(rnn:maskZero(1)))

    	--batch norm
    	--utterance_rnn:add(nn.Sequencer(nn.BatchNormalization(opt.enc_hidden_size)))

	if opt.drop_rate > 0 then
		utterance_rnn:add(nn.Sequencer(nn.Dropout(opt.drop_rate)))
	end

	utterance_rnn:add(nn.Select(1, -1))

	return utterance_rnn
end

--build hred encoder(utterance encoder and context encoder)
function build_hred_encoder(recurrence)
	local hred_enc = nn.Sequential()
	local hred_enc_rnn
	local par = nn.ParallelTable()
	
	--build parallel utterance rnns
	local rnns = {}
	for i = 1, 2 do
		table.insert(rnns, RNN_elem(recurrence))
	end

	rnns[2] = rnns[1]:clone('weight', 'bias', 'gradWeight', 'gradBias') --utterance 2 rnn share the weight with utterance 1 rnn

	for i = 1, 2 do
		par:add(rnns[i])
	end

	hred_enc:add(par)
	
	hred_enc:add(nn.JoinTable(1, 2))
	hred_enc:add(nn.Reshape(2, opt.batch_size, opt.enc_hidden_size))

	--build context layer
	local context_layer = nn.Sequential()
	hred_enc_rnn = recurrence(opt.enc_hidden_size, opt.context_hidden_size)
	context_layer:add(nn.Sequencer(hred_enc_rnn:maskZero(1)))

    	--batch norm
    	--context_layer:add(nn.Sequencer(nn.BatchNormalization(opt.context_hidden_size)))
    
	if opt.drop_rate > 0 then
		context_layer:add(nn.Sequencer(nn.Dropout(opt.drop_rate)))
	end

	context_layer:add(nn.Select(1, -1))
	hred_enc:add(context_layer)

	return hred_enc, hred_enc_rnn
end

--build topic and policy net
function build_topic_policy_net(recurrence)
	local topic_net = nn.Sequential()
	local policy_net = nn.Sequential()

	local topic_net_linear = nn.Linear(opt.context_hidden_size, opt.topic_num_size)
	topic_net:add(topic_net_linear)
	topic_net:add(nn.SoftMax())

	local policy_net_rnn = recurrence(opt.word_dim+opt.topic_num_size, opt.dec_hidden_size)
	policy_net:add(nn.Sequencer(policy_net_rnn:maskZero(1)))

	if opt.drop_rate > 0 then
		policy_net:add(nn.Sequencer(nn.Dropout(opt.drop_rate)))
	end

	local policy_net_linear = nn.Linear(opt.dec_hidden_size, opt.vocab_size)
	policy_net:add(nn.Sequencer(nn.MaskZero(policy_net_linear, 1)))
	policy_net:add(nn.Sequencer(nn.MaskZero(nn.SoftMax(), 1)))

	return topic_net, policy_net, policy_net_rnn
end

--build model 
function build()
	local recurrence = nn[opt.cell]
	print('Building model...')
	print('Layer type: '..opt.cell)
	print('Vocab size: '..opt.vocab_size)
	print('Embedding size: '..opt.word_dim)
	print('Encoder layer hidden size: '..opt.enc_hidden_size)
	print('Context layer hidden size: '..opt.context_hidden_size)
	print('Decoder layer hidden size: '..opt.dec_hidden_size)
	print('Topic num size: '..opt.topic_num_size)
	print('Top k actions: '..opt.top_k_actions)
	
	--my criterion
	--note: don't need topic_net criterion, can directly cal the gradients of the topic net
	local policy_net_criterion = nn.MySequencerCriterion(nn.myCriterion()) 

	local hred_enc, hred_enc_rnn, topic_net, policy_net, policy_net_rnn
	
	-- whether to load pre-trained model from load_model_file
	if opt.load_model == 0 then
		hred_enc, hred_enc_rnn = build_hred_encoder(recurrence)
		topic_net, policy_net, policy_net_rnn = build_topic_policy_net(recurrence)
		
	else
		--load the trained model
		assert(path.exists(opt.load_model_file), 'check the model file path')
		print('Loading model from: '..opt.load_model_file..'...')
		local model_and_opts = torch.load(opt.load_model_file)
		local model, model_opt = model_and_opts[1], model_and_opts[2]
		
		--load the model components
		hred_enc = model[1]:double()
		hred_enc_rnn = model[2]:double()
		topic_net = model[3]:double()
		policy_net = model[4]:double()
		policy_net_rnn = model[5]:double()
		-- if the batch size is changed
		hred_enc:remove(3)
		hred_enc:insert(nn.Reshape(2, opt.batch_size, opt.enc_hidden_size), 3)
	end
	
	local layers = {hred_enc, topic_net, policy_net}
	
	--run on GPU
	if opt.gpu_id >= 0 then
		for i = 1, #layers do
			layers[i]:cuda()
		end
		policy_net_criterion:cuda()
	end
	local Model = nn.Sequential()
	Model:add(hred_enc)
	Model:add(topic_net)
	Model:add(policy_net)
	local params, grad_params = Model:getParameters()

	if opt.gpu_id >= 0 then
		params:cuda()
		grad_params:cuda()
	end

	--package model for training
	local model = {
		hred_enc,
		hred_enc_rnn,
		topic_net,
		policy_net,
		policy_net_rnn,
		params,
		grad_params
	}
	
	print('Building model successfully...')
	return model, policy_net_criterion
end

