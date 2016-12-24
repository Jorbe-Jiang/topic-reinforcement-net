--------------------------------------------------
--some extra functions
--------------------------------------------------
function idx_2_word()
	local vocab_file = './datas/VOCAB.dat'
	local idx_2_word = {}
	for line in io.lines(vocab_file) do
		table.insert(idx_2_word, line)
	end
	return idx_2_word
end

function word_2_idx()
	local vocab_file = './datas/VOCAB.dat'
	local word_2_idx = {}
	local word_idx = 1
	for line in io.lines(vocab_file) do
		word_2_idx[line] = word_idx
		word_idx = word_idx + 1
	end
	return word_2_idx
end

function forward_connect(hred_enc_rnn, policy_net_rnn)
	policy_net_rnn.userPrevOutput = nn.rnn.recursiveCopy(policy_net_rnn.userPrevOutput, hred_enc_rnn.outputs[2])
	if opt.cell ~= 'GRU' then
		policy_net_rnn.userPrevCell = nn.rnn.recursiveCopy(policy_net_rnn.userPrevCell, hred_enc_rnn.cells[2])
	end
end

function backward_connect(hred_enc_rnn, policy_net_rnn)
	if opt.cell ~= 'GRU' then
		hred_enc_rnn.userNextGradCell = nn.rnn.recursiveCopy(hred_enc_rnn.userNextGradCell, policy_net_rnn.userGradPrevCell)
	end
	hred_enc_rnn.gradPrevOutput = nn.rnn.recursiveCopy(hred_enc_rnn.gradPrevOutput, policy_net_rnn.userGradPrevOutput)
end

local idx_2_word_table = idx_2_word()
-- U3s_words_idx:[n_U3s, seqLen]
function translate_to_word(U3s_words_idx)
	local U3s_words_table = {}
	local n_U3s = U3s_words_idx:size(1)
	local seq_len = U3s_words_idx:size(2)
	for i = 1, n_U3s do
		local U3_words = {}
		for j = 1, seq_len do
			if U3s_words_idx[{i, j}] ~= 1 then
				local word = idx_2_word_table[U3s_words_idx[{i, j}]]
				table.insert(U3_words, word)
			end
		end
		table.insert(U3s_words_table, U3_words)
	end
	return U3s_words_table
end

--cal gram_pros p(n_history_actions, action)
function cal_grams(N_gram_mat, n_history_actions, t_action)
	local gram_pros
	
	gram_pros = torch.Tensor(opt.batch_size, opt.top_k_actions):fill(1.0)
	
	local n_steps = #n_history_actions  --n_step <= 2

	for i = 1, opt.batch_size do
		for j = 1, n_steps do
			if torch.sum(t_action[i]) > 0 then
				for k = 1, opt.top_k_actions do
					if j == n_steps then
						gram_pros[{i, k}] = gram_pros[{i, k}] * N_gram_mat[{n_history_actions[j][{i, k}], t_action[{i, k}]}]
					else
						gram_pros[{i, k}] = gram_pros[{i, k}] * N_gram_mat[{n_history_actions[j][{i, k}], n_history_actions[j+1][{i, k}]}]
					end
				end
			end
		end
	end
	return gram_pros
end

--cal reward according to action and history_actions
function cal_reward(N_gram_mat, t_action, n_history_actions, true_word_idx)
	--[[
	if action == true_word_idx
		if p(n_history_actions, action) > opt.alpha
			reward = 10.0
		else
			reward = 1.0
	-- action != true_word_idx
	else 
		if the true_word_idx == 'end'
			if p(n_history_actions, action) > opt.alpha
				reward = -1.0
			else
				reward = -10.0
		elif p(n_history_actions, action) > opt.alpha
			reward = 1.0
		else
			reward = -10.0
	]]--
	local t_reward
	t_reward = torch.Tensor(opt.batch_size, opt.top_k_actions):fill(0.0)
	--t_reward = torch.Tensor(1, opt.batch_size):fill(0.0)

	--gram_pros:[batch_size, top_k_actions]
	gram_pros = cal_grams(N_gram_mat, n_history_actions, t_action)

	for i = 1, opt.batch_size do
		for j = 1, opt.top_k_actions do
			if t_action[{i, j}] == true_word_idx[i] then
				if gram_pros[{i, j}] > opt.alpha then
					t_reward[{i, j}] = 10.0
				else
					t_reward[{i, j}] = 1.0
				end
			else
				if true_word_idx[i] == 2 then
					if gram_pros[{i, j}] > opt.alpha then
						t_reward[{i, j}] = -1.0
					else
						t_reward[{i, j}] = -10.0
					end
				elseif gram_pros[{i, j}] > opt.alpha then
					t_reward[{i, j}] = 1.0
				else
					t_reward[{i, j}] = -10.0
				end
			end
		end
	end

	local zero_mask = torch.eq(true_word_idx, 0)
	for i = 1, opt.batch_size do
		if zero_mask[i] == 1 then
			t_reward[i] = 0
		end
	end	
	
	return t_reward
end

--generate action and reward according to policy_net outputs and history actions
function gen_action_and_reward(N_gram_mat, pred_outs, n_history_actions, true_word_idx)
	local t_action, t_reward

	t_action = torch.Tensor(opt.batch_size, opt.top_k_actions):fill(0):int()
		
	local seed = torch.rand(1)[1]
	
	if seed < opt.epsilon then
		for i = 1, opt.batch_size do
			for j = 1, opt.top_k_actions do
				t_action[{i, j}] = torch.random(opt.vocab_size)
			end
		end
	elseif seed > opt.epsilon and seed < 0.5 then
		t_action[{{}, opt.top_k_actions}] = true_word_idx
		if opt.top_k_actions > 1 then
			for i = 1, opt.batch_size do
				for j = 1, opt.top_k_actions-1 do
					t_action[{i, j}] = torch.random(opt.vocab_size)
				end
			end
		end
	else
		local max_values, max_idx = torch.topk(pred_outs, opt.top_k_actions, 2, true)
		t_action = max_idx:int()
	end

	local zero_mask = torch.eq(true_word_idx, 0)

	for i = 1, opt.batch_size do
		if zero_mask[i] == 1 then
			t_action[i] = 0
		end
	end
	
	t_reward = cal_reward(N_gram_mat, t_action, n_history_actions, true_word_idx)

	return t_action, t_reward
end

--q_ts:[seqLen, batch_size, vocab_size] , actions:[seqLen, batch_size, top_k_actions], policy_net_rewards: [seqLen, batch_size, top_k_actions]
function get_values(N_gram_mat, dec_output, dec_tar_outputs_2)
	local q_ts, actions, policy_net_rewards
	
	local seq_len = dec_tar_outputs_2:size(1)
		
	q_ts = torch.Tensor(seq_len, opt.batch_size, opt.vocab_size):fill(0.0)
	actions = torch.Tensor(seq_len, opt.batch_size, opt.top_k_actions):fill(0):int()
	policy_net_rewards = torch.Tensor(seq_len, opt.batch_size, opt.top_k_actions):fill(0.0)
	

	local n_history_actions = {torch.Tensor(opt.batch_size, opt.top_k_actions):fill(1):int(),}
	local true_word_idx 
	
	for time_step = 1, seq_len do
		true_word_idx = dec_tar_outputs_2[time_step]
		t_action, t_reward = gen_action_and_reward(N_gram_mat, dec_output[time_step], n_history_actions, true_word_idx)
		if #n_history_actions > 2 then --only store prev 2 time steps history actions
			table.remove(n_history_actions, 1)
		else
			table.insert(n_history_actions, t_action:view(opt.batch_size, opt.top_k_actions))
		end
	
		actions[time_step] = t_action
		policy_net_rewards[time_step] = t_reward
	end

	for i = seq_len, 1, -1 do
		for j = 1, opt.batch_size do
			if torch.sum(actions[{i, j}]) > 0 then
				for k = 1, opt.top_k_actions do
					local t_action = actions[{i, j, k}]
					if i == seq_len then
						q_ts[{i, j, t_action}] = policy_net_rewards[{i, j, k}]
					else
						local next_t_action = actions[{i+1, j, k}]
						if next_t_action == 0 then
							q_ts[{i, j, t_action}] = policy_net_rewards[{i, j, k}]
						else
							q_ts[{i, j, t_action}] = policy_net_rewards[{i, j, k}] + opt.gamma * q_ts[{i+1, j, next_t_action}]
						end	
					end
				end
			end
		end
	end

	return q_ts, actions, policy_net_rewards
end

--calculate topic_net loss
--topic_net_outputs:[batch_size, topic_num_size] q_ts:[seqLen, batch_size, vocab_size]
function cal_topic_net_loss(topic_net_outputs, q_ts)
	local topic_net_loss = 0.0
	local seq_len = q_ts:size(1)

	q_ts = torch.permute(q_ts, 2, 1, 3) --[batch_size, seqLen, vocab_size]

	for i = 1, opt.batch_size do
		--local p_value = torch.sum(torch.log(topic_net_outputs[i]+opt.eps*1e-4))
		--topic_net_loss = topic_net_loss - p_value * topic_net_rewards
		local topic_net_rewards = torch.sum(q_ts[i])/opt.top_k_actions
		topic_net_loss = topic_net_loss - topic_net_rewards
	end

	--sizeAverage
	topic_net_loss = topic_net_loss / opt.batch_size

	return topic_net_loss
end

--calculate the topic net gradient
--have some problems:for every sample it has same gradient with every unit in the last layer(topic num size)
function cal_topic_net_gradient(topic_net_outputs, q_ts)	
	local topic_net_gradient = torch.Tensor(opt.batch_size, opt.topic_num_size)

	q_ts = torch.permute(q_ts, 2, 1, 3) --[batch_size, seqLen, vocab_size]

	for i = 1, opt.batch_size do
		local p_value = topic_net_outputs[i]:double()
		local topic_net_rewards = torch.sum(q_ts[i])/opt.top_k_actions
		topic_net_gradient[i] = -torch.Tensor(opt.topic_num_size):fill(topic_net_rewards):cdiv(p_value+opt.eps*1e-4)
		--topic_net_gradient[i] = -torch.Tensor(opt.topic_num_size):fill(topic_net_rewards) --if the above gradient is too big, then change to this fumula
		--print(topic_net_gradient[i])
		--os.exit()
	end

	--sizeAverage
	topic_net_gradient = topic_net_gradient / opt.batch_size

	return topic_net_gradient
end


--------------------------------------------------
--training process
--------------------------------------------------
--training
function train(model, policy_net_criterion, batches_train_data, batches_valid_data)
	print('Begin to train...')
	
	--hred_enc_inputs:<U1, U2> {Tensor(U1_seqLen, batchSize), Tensor(U2_seqLen, batchSize)}
	--dec_inputs:<U2, U3> {Tensor(U2_seqLen, batchSize), Tensor(U3_seqLen, batchSize)}
	--dec_tar_outputs:<U2, U3> {Tensor(U2_seqLen, batchSize), Tensor(U3_seqLen, batchSize)}
	local hred_enc_inputs, dec_inputs, dec_tar_outputs
	
	local hred_enc_output, policy_net_inputs, dec_output 

	--model components
	local hred_enc, hred_enc_rnn, topic_net, policy_net, policy_net_rnn, params, grad_params

	hred_enc = model[1]
	hred_enc_rnn = model[2]
	topic_net = model[3]
	policy_net = model[4]
	policy_net_rnn = model[5]
	params = model[6]
	grad_params = model[7]

	--topic_net_outputs:[batch_size, opt.topic_num_size] policy_net_rewards: [seqLen, batch_size, top_k_actions]
	-- q_ts: [seqLen, batch_size, vocab_size], actions: [seqLen, batch_size, top_k_actions]
	local topic_net_outputs, policy_net_rewards, q_ts, actions

	local policy_net_emb_layer = nn.LookupTableMaskZero(opt.vocab_size, opt.word_dim):setMaxNorm(2)
	if opt.gpu_id >= 0 then
        	policy_net_emb_layer = policy_net_emb_layer:cuda()
    	end	
	policy_net_emb_layer:share(hred_enc:get(1):get(1):get(1), 'weight', 'bias')
    	
	local N_gram_mat = npy4th.loadnpy(opt.load_n_gram_file)
	--print(N_gram_mat:size())
	--os.exit()

	--forward process
	function topic_net_forward()
		hred_enc_output = hred_enc:forward(hred_enc_inputs)
		topic_net_outputs = topic_net:forward(hred_enc_output)
		collectgarbage()
	end

	function policy_net_forward()
		hred_enc_output = hred_enc:forward(hred_enc_inputs)
		forward_connect(hred_enc_rnn, policy_net_rnn)

		--[seqLen, batch_size, word_dim]
		local policy_net_words_vec = policy_net_emb_layer:forward(dec_inputs[2])
		local cp_topic_net_outputs = torch.Tensor(opt.batch_size, opt.topic_num_size):copy(topic_net_outputs)
		local seq_len = dec_inputs[2]:size(1)
		
        	if opt.gpu_id >= 0 then
		    	policy_net_inputs = torch.CudaTensor(seq_len, opt.batch_size, opt.word_dim + opt.topic_num_size)
			cp_topic_net_outputs = cp_topic_net_outputs:cuda()
        	else
		    	policy_net_inputs = torch.Tensor(seq_len, opt.batch_size, opt.word_dim + opt.topic_num_size)
        	end

		for i = 1, seq_len do
            		if opt.gpu_id >= 0 then
			    	policy_net_inputs[i] = torch.CudaTensor.cat(policy_net_words_vec[i], cp_topic_net_outputs)
            		else
			    	policy_net_inputs[i] = torch.Tensor.cat(policy_net_words_vec[i], cp_topic_net_outputs)
            		end
		end

		dec_output = policy_net:forward(policy_net_inputs)
		dec_output = dec_output:double()
		collectgarbage()
	end

	--update topic net
	function topic_net_eval(x)
		assert(x == params)
		if x == nil then
			print('Notes: params is nil!!!')
			os.exit()
		end
		
		--reset params gradients
		grad_params:zero()

		--calculate topic_net loss and gradient
		--local topic_net_loss = cal_topic_net_loss(topic_net_outputs, policy_net_rewards)
		--local topic_net_gradient = cal_topic_net_gradient(topic_net_outputs, policy_net_rewards)
		local topic_net_loss = cal_topic_net_loss(topic_net_outputs, q_ts)
		local topic_net_gradient = cal_topic_net_gradient(topic_net_outputs, q_ts)

		--[[
		if opt.grad_clip > 0 then 
			topic_net_gradient:clamp(-opt.grad_clip, opt.grad_clip)
		end
		]]--
		
		if opt.gpu_id >= 0 then
			topic_net_gradient = topic_net_gradient:cuda()
		end

		local hred_enc_grad_input = topic_net:backward(hred_enc_output, topic_net_gradient)
		--print(torch.sum(hred_enc_grad_input))

		hred_enc:backward(hred_enc_inputs, hred_enc_grad_input)

		if opt.grad_clip > 0 then 
			grad_params:clamp(-opt.grad_clip, opt.grad_clip)
		end
		
		return topic_net_loss, grad_params
	end

	--update policy net
	function policy_net_eval(x)
		assert(x == params)
		if x == nil then
			print('Notes: params is nil!!!')
			os.exit()
		end
		
		--reset params gradients
		grad_params:zero()

		local seq_len = dec_inputs[2]:size(1)
		
		local f_inputs = torch.Tensor(seq_len, opt.batch_size, opt.vocab_size):fill(0.0)
		for i = 1, seq_len do
			for j = 1, opt.batch_size do
				if torch.sum(actions[{i, j}]) > 0 then
					for k = 1, opt.top_k_actions do
						local action = actions[{i, j, k}]
						local f_input = dec_output[{i, j, action}]
						f_inputs[{i, j, action}] = f_input
					end
				end
			end
		end
		
		if opt.gpu_id >= 0 then
			f_inputs = f_inputs:cuda()
			--q_ts = q_ts:cuda()
		end
		local policy_net_loss = policy_net_criterion:forward(f_inputs, dec_tar_outputs[2], q_ts)
		local policy_net_grad_output = policy_net_criterion:backward(f_inputs, dec_tar_outputs[2], q_ts)

		--[[
		if opt.grad_clip > 0 then 
			policy_net_grad_output:clamp(-opt.grad_clip, opt.grad_clip)
		end
		]]--
		
		if opt.gpu_id >= 0 then
			policy_net_grad_output = policy_net_grad_output:cuda()
		end

		policy_net:backward(policy_net_inputs, policy_net_grad_output)
		backward_connect(hred_enc_rnn, policy_net_rnn)

		local zero_tensor = opt.gpu_id >= 0 and torch.CudaTensor(hred_enc_output):zero() or torch.Tensor(hred_enc_output):zero()
		hred_enc:forward(hred_enc_inputs)
		hred_enc:backward(hred_enc_inputs, zero_tensor) 
		
		if opt.grad_clip > 0 then grad_params:clamp(-opt.grad_clip, opt.grad_clip) end

		return policy_net_loss, grad_params
	end
	

	--evaluate policy net loss and topic net loss
	function eval_loss(x)
		assert(x == params)
		if x == nil then
			print('Notes: params is nil!!!')
			os.exit()
		end

		hred_enc_output = hred_enc:forward(hred_enc_inputs)
		forward_connect(hred_enc_rnn, policy_net_rnn)
			
		--[seqLen, batch_size, word_dim]
		local policy_net_words_vec = policy_net_emb_layer:forward(dec_inputs[2])
		local seq_len = dec_inputs[2]:size(1)
		topic_net_outputs = topic_net:forward(hred_enc_output)
		local cp_topic_net_outputs = torch.Tensor(opt.batch_size, opt.topic_num_size):copy(topic_net_outputs)
		
        	if opt.gpu_id >= 0 then
		    	policy_net_inputs = torch.CudaTensor(seq_len, opt.batch_size, opt.word_dim + opt.topic_num_size)
			cp_topic_net_outputs = cp_topic_net_outputs:cuda()
        	else
		    	policy_net_inputs = torch.Tensor(seq_len, opt.batch_size, opt.word_dim + opt.topic_num_size)
        	end

		for i = 1, seq_len do
            		if opt.gpu_id >= 0 then
			    	policy_net_inputs[i] = torch.CudaTensor.cat(policy_net_words_vec[i], cp_topic_net_outputs)
            		else
			    	policy_net_inputs[i] = torch.Tensor.cat(policy_net_words_vec[i], cp_topic_net_outputs)
            		end
		end

		dec_output = policy_net:forward(policy_net_inputs)
		dec_output = dec_output:double()
		collectgarbage()
	
		--q_ts:[seqLen, batch_size, vocab_size] , actions:[seqLen, batch_size, top_k_actions], policy_net_rewards: [seqLen, batch_size, top_k_actions]
		q_ts, actions, policy_net_rewards = get_values(N_gram_mat, dec_output, dec_tar_outputs[2])
		
		local seq_len = dec_inputs[2]:size(1)

		--local f_inputs = torch.Tensor(seq_len, opt.batch_size, opt.top_k_actions):fill(0.0) 
		local f_inputs = torch.Tensor(seq_len, opt.batch_size, opt.vocab_size):fill(0.0)
		for i = 1, seq_len do
			for j = 1, opt.batch_size do
				if torch.sum(actions[{i, j}]) > 0 then
					for k = 1, opt.top_k_actions do
						local action = actions[{i, j, k}]
						local f_input = dec_output[{i, j, action}]
						f_inputs[{i, j, action}] = f_input
					end
				end
			end
		end
		
		if opt.gpu_id >= 0 then
			f_inputs = f_inputs:cuda()
			--q_ts = q_ts:cuda()
		end
		local policy_net_loss = policy_net_criterion:forward(f_inputs, dec_tar_outputs[2], q_ts)
		local topic_net_loss = cal_topic_net_loss(topic_net_outputs, q_ts)

		local norm_policy_net_loss = policy_net_loss*opt.batch_size/torch.sum(torch.gt(dec_inputs[2],0))
		local norm_topic_net_loss = topic_net_loss*opt.batch_size/torch.sum(torch.gt(dec_inputs[2],0))
		return norm_policy_net_loss, norm_topic_net_loss
	end

	function argmax(pred_outs)
		max_values, max_idxs = torch.max(pred_outs, 3)
		return max_idxs
	end

	function print_pred_results()
		local U3s_pred_pros = dec_output
		local U3s_pred_idx = argmax(U3s_pred_pros)
		U3s_pred_idx = U3s_pred_idx:view(U3s_pred_idx:size(1), opt.batch_size):t()
		if opt.gpu_id >= 0 then
			U3s_pred_idx = U3s_pred_idx:cudaLong()
			U3s_true_idx = dec_tar_outputs[2]:t():cudaLong()
		else
			U3s_pred_idx = U3s_pred_idx:long()
			U3s_true_idx = dec_tar_outputs[2]:t():long()		
		end

		local print_U3s_pred_words_table = translate_to_word(U3s_pred_idx:narrow(1, 1, 3))
		local print_U3s_true_words_table = translate_to_word(U3s_true_idx:narrow(1, 1, 3))
		print('U3s pred(3 arrow):')
		--print(U3s_pred_idx:narrow(1, 1, 3))
		for i = 1, #print_U3s_pred_words_table do
			print(i..'> '..table.concat(print_U3s_pred_words_table[i], ' '))
		end
		print('U3s true(3 arrow):')
		--print(U3s_true_idx:narrow(1, 1, 3))
		for i = 1, #print_U3s_true_words_table do
			print(i..'. '..table.concat(print_U3s_true_words_table[i], ' '))
		end
		print(U3s_pred_idx:ne(U3s_true_idx):sum())
	end			

	local U1s_batches_enc, U2s_batches_enc, U2s_batches_dec, U3s_batches_dec, U2s_batches_tar, U3s_batches_tar, nbatches
	U1s_batches_enc = batches_train_data[1]
	U2s_batches_enc = batches_train_data[2]
	U2s_batches_dec = batches_train_data[3]
	U3s_batches_dec = batches_train_data[4]
	U2s_batches_tar = batches_train_data[5]
	U3s_batches_tar = batches_train_data[6]
	nbatches = batches_train_data[7]
	
	--for evaluate
	local valid_U1s_batches_enc, valid_U2s_batches_enc, valid_U2s_batches_dec, valid_U3s_batches_dec, valid_U2s_batches_tar, valid_U3s_batches_tar, valid_nbatches
	valid_U1s_batches_enc = batches_valid_data[1]
	valid_U2s_batches_enc = batches_valid_data[2]
	valid_U2s_batches_dec = batches_valid_data[3]
	valid_U3s_batches_dec = batches_valid_data[4]
	valid_U2s_batches_tar = batches_valid_data[5]
	valid_U3s_batches_tar = batches_valid_data[6]
	valid_nbatches = batches_valid_data[7]

	local date = os.date("%m_%d") --today's date
	local policy_net_losses = {}
	local topic_net_losses = {}
	local eval_policy_net_losses = {}
	local eval_topic_net_losses = {}
	local prev_eval_loss, curr_eval_loss
	local optim_state = {learningRate = opt.lr, weightDecays = 1e-2, momentum = 1e-4}
	local start_time = os.time() --begin time
	local cur_mean_policy_net_losses = 0.0
	local cur_mean_topic_net_losses = 0.0

	for i = 1, opt.num_epochs do
		hred_enc:training()
		topic_net:training()
		policy_net:training()
		local batches_order = torch.randperm(nbatches) --shuffle the order of all batches data

		if (i >= opt.start_decay_at or opt.start_decay == 1) and opt.lr > 0.001 then
			opt.lr = opt.lr * opt.lr_decay_rate
			optim_state.learningRate = opt.lr
		end

		local epoch_policy_net_losses = {}
		local epoch_topic_net_losses = {}
		
		for j = 1, nbatches do
			hred_enc_inputs = {U1s_batches_enc[batches_order[j]], U2s_batches_enc[batches_order[j]]}
			dec_inputs = {U2s_batches_dec[batches_order[j]], U3s_batches_dec[batches_order[j]]}
			dec_tar_outputs = {U2s_batches_tar[batches_order[j]], U3s_batches_tar[batches_order[j]]}

			if opt.gpu_id >= 0 then
				for i = 1, 2 do
					hred_enc_inputs[i] = hred_enc_inputs[i]:int():cuda()
				end
				dec_inputs[2] = dec_inputs[2]:int():cuda()
			end

			topic_net_forward()
			policy_net_forward()
			--q_ts:[seqLen, batch_size, vocab_size] , actions:[seqLen, batch_size, top_k_actions], policy_net_rewards: [seqLen, batch_size, top_k_actions]
			q_ts, actions, policy_net_rewards = get_values(N_gram_mat, dec_output, dec_tar_outputs[2])

			--update topic net 
			local _1, topic_net_loss = optim[opt.optimizer](topic_net_eval, params, optim_state)
			collectgarbage()
			--update policy net
			local _2, policy_net_loss = optim[opt.optimizer](policy_net_eval, params, optim_state)
			collectgarbage()

			local norm_policy_net_loss = policy_net_loss[1]*opt.batch_size/torch.sum(torch.gt(dec_inputs[2],0))
			local norm_topic_net_loss = topic_net_loss[1]*opt.batch_size/torch.sum(torch.gt(dec_inputs[2],0))

			if norm_policy_net_loss < -0.5 or i % opt.print_every == 0 then
				print_pred_results()
			end

			epoch_policy_net_losses[j] = norm_policy_net_loss
			epoch_topic_net_losses[j] = norm_topic_net_loss

			local msg = string.format("Epoch: %d, complete: %.3f%%, lr = %.4f, norm_policy_net_loss = %6.4f, norm_topic_net_loss = %6.4f, grad norm = %6.2e ", i, 100*j/nbatches, opt.lr, norm_policy_net_loss, norm_topic_net_loss, torch.norm(grad_params))
			io.stdout:write(msg..'\b\r')
			io.flush()
		end

		cur_mean_policy_net_losses = torch.mean(torch.Tensor(epoch_policy_net_losses))
		cur_mean_topic_net_losses = torch.mean(torch.Tensor(epoch_topic_net_losses))
		
		table.insert(policy_net_losses, cur_mean_policy_net_losses)
		table.insert(topic_net_losses, cur_mean_topic_net_losses)

		if i % opt.eval_every == 0 or i == opt.num_epochs then
			print('Evaluating ...')
			
			for k = 1, valid_nbatches do
				hred_enc:evaluate()
				topic_net:evaluate()
				policy_net:evaluate()
				hred_enc_inputs = {valid_U1s_batches_enc[k], valid_U2s_batches_enc[k]}
				dec_inputs = {valid_U2s_batches_dec[k], valid_U3s_batches_dec[k]}
				dec_tar_outputs = {valid_U2s_batches_tar[k], valid_U3s_batches_tar[k]}

				if opt.gpu_id >= 0 then
					for i = 1, 2 do
						hred_enc_inputs[i] = hred_enc_inputs[i]:int():cuda()
					end
					dec_inputs[2] = dec_inputs[2]:int():cuda()
				end

				local norm_policy_net_loss, norm_topic_net_loss = eval_loss(params)
				collectgarbage()
				print("eval policy net rewards: ", -norm_policy_net_loss)
				print("eval topic net rewards: ", -norm_topic_net_loss)
				table.insert(eval_policy_net_losses, norm_policy_net_loss)
				table.insert(eval_topic_net_losses, norm_topic_net_loss)
			end
			
			local mean_loss = torch.mean(torch.Tensor(eval_policy_net_losses))
			
			if prev_eval_loss == nil then
				prev_eval_loss = mean_loss
				curr_eval_loss = mean_loss
			else
				curr_eval_loss = mean_loss
			end

			--early stoping
			if prev_eval_loss > 0 then
                		if curr_eval_loss > prev_eval_loss * 3 then
                    			print('Loss is exploding, early stoping...')
                    			os.exit()
                		end
            		end

            		if prev_eval_loss < 0 then
				if curr_eval_loss > prev_eval_loss/3 then
					print('Loss is exploding, early stoping...')
					os.exit()
				end
            		end

			--start decay learning rate
			if curr_eval_loss > prev_eval_loss then
				opt.start_decay = 1
			else
				opt.start_decay = 0
			end
		
			prev_eval_loss = curr_eval_loss		

			local save_file = string.format("./model/%s_epoch_%d_epochs_%d_%.2f_model.t7", date, i, opt.num_epochs, mean_loss)
			print('Saving model to: ', save_file)
			hred_enc:clearState()
			hred_enc_rnn:clearState()
			topic_net:clearState()
			policy_net:clearState()
			policy_net_rnn:clearState()
			local package_model = {
				hred_enc:double(), --hred_enc
				hred_enc_rnn:double(), --hred_enc_rnn
				topic_net:double(), --topic_net
				policy_net:double(), --policy_net
				policy_net_rnn:double()  --policy_net_rnn
			}
			torch.save(save_file, {package_model, opt})
			local tmp_result_file_1 = string.format("./results/%s_epoch_%d_epochs_%d_%d_tmp_policy_net_result.npy", date, i,  opt.num_epochs, opt.batch_size)
			print('Saving tmp results to: ', tmp_result_file_1)
			npy4th.savenpy(tmp_result_file_1, torch.Tensor(policy_net_losses))
			local tmp_result_file_2 = string.format("./results/%s_epoch_%d_epochs_%d_%d_tmp_topic_net_result.npy", date, i,  opt.num_epochs, opt.batch_size)
			print('Saving tmp results to: ', tmp_result_file_2)
			npy4th.savenpy(tmp_result_file_2, torch.Tensor(topic_net_losses))
			print('Evaluating end ...')
			eval_policy_net_losses = {}
			eval_val_net_losses = {}
			if opt.gpu_id >= 0 then
				hred_enc:cuda()
				topic_net:cuda()
				policy_net:cuda()
				hred_enc_rnn:cuda()
				policy_net_rnn:cuda()
			end
		end
		collectgarbage()
	end
	--saving results
	local result_file_1 = string.format("./results/%s_%d_%d_policy_net_result.npy", date, opt.num_epochs, opt.batch_size)
	print('Saving results to: ', result_file_1)
	npy4th.savenpy(result_file_1, torch.Tensor(policy_net_losses))
	local result_file_2 = string.format("./results/%s_%d_%d_topic_net_result.npy", date, opt.num_epochs, opt.batch_size)
	print('Saving results to: ', result_file_2)
	npy4th.savenpy(result_file_2, torch.Tensor(topic_net_losses))
	local end_time = os.time()
	local train_time = os.difftime(end_time, start_time)
	print("Training cost :", string.format("%.2d:%.2d:%.2d", train_time/(60*60), train_time/60%60, train_time%60))
	print('Training end ...')
end	
