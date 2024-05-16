import torch
import numpy as np

from ..builder import LOSSES

@LOSSES.register_module()
class BucketedRankSort(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets,delta_RS=0.50): 
        
        grad = torch.zeros(targets.shape).cuda()
        metric = torch.zeros(0).cuda()
        
        #Store the original indices of logits and targets
        old_targets = targets
        p_indices = torch.nonzero(old_targets > 0.).flatten()
        n_indices = torch.nonzero(old_targets == 0).flatten()

        #If no positive logit, return zero loss
        if len(p_indices) == 0:
            ctx.save_for_backward(grad)
            return torch.zeros(1).cuda(), torch.zeros(1).cuda()

        
        p_and_n_indices = torch.cat((p_indices, n_indices))
        p_and_n_logits = logits[p_and_n_indices]
        p_and_n_targets = targets[p_and_n_indices]

        #Sort the logits and targets
        sorted_p_and_n_logits, sorted_p_and_n_indices = torch.sort(p_and_n_logits, descending=True)
        sorted_p_and_n_targets = p_and_n_targets[sorted_p_and_n_indices]


        f_indices = sorted_p_and_n_targets > 0.
        f_logits = sorted_p_and_n_logits[f_indices]
        f_targets = sorted_p_and_n_targets[f_indices]

        threshold_logit = torch.min(f_logits) - delta_RS

        #Get rid of irrelevant background logits
        irrelevant_b_index = torch.nonzero(sorted_p_and_n_logits >= threshold_logit)[-1] + 1
        #Get relevant logits and targets
        relevant_logits = sorted_p_and_n_logits[0:irrelevant_b_index]
        relevant_targets = sorted_p_and_n_targets[0: irrelevant_b_index]
        #Split into foreground and background
        relevant_f_indices = torch.nonzero(relevant_targets > 0.)
        relevant_b_indices = torch.nonzero(relevant_targets == 0)

        #If no relevant targets, return zero loss
        if len(relevant_targets) == 0 or len(relevant_f_indices) == 0:
            ctx.save_for_backward(grad)
            return torch.zeros(0).to(logits.device), torch.zeros(1).to(logits.device)
        
        fg_num = len(f_targets)

        #If relevant background logits exist, calculate buckets and their sizes 
        if len(relevant_b_indices) != 0:
            #Calculate background buckets and their sizes
            bucket_sizes_b = (torch.sub(relevant_f_indices[1:], relevant_f_indices[:len(relevant_f_indices) - 1]) - 1)
            bucket_sizes_b = torch.cat((torch.tensor([[relevant_f_indices[0]]], device=logits.device), bucket_sizes_b))

            if relevant_f_indices[-1] < relevant_b_indices[-1]:
                bucket_sizes_b = torch.cat((bucket_sizes_b, torch.tensor([[relevant_b_indices[-1] -  relevant_f_indices[-1]]], device=logits.device)))
            
            bucket_sizes_b = bucket_sizes_b[(bucket_sizes_b != 0)]
            bucket_sizes_b = bucket_sizes_b.reshape((len(bucket_sizes_b), 1))
            l_bucket_sizes_b = [int(x) for x in torch.Tensor.tolist(bucket_sizes_b.flatten())]

            relevant_bg_buckets = torch.split(relevant_logits[relevant_b_indices], tuple(l_bucket_sizes_b))
            bg_bucket_mean = torch.Tensor([torch.mean(bucket) for bucket in list(relevant_bg_buckets)]).to(logits.device)

            #No extra bucketing for foreground objects, so their bucket sizes are 1
            bucket_sizes_f = torch.ones(len(f_logits)).to(logits.device)
            all_buckets = torch.cat((f_logits, bg_bucket_mean))

            #Sort all buckets
            all_buckets, indices_s = torch.sort(all_buckets, descending=True)

            #Initialize new targets
            new_targets = torch.zeros(len(all_buckets)).to(logits.device)
            new_targets[indices_s < len(f_logits)] = sorted_p_and_n_targets[sorted_p_and_n_targets != 0]


            if torch.max(new_targets) <= 0:
                return grad, metric

            labels_p = (new_targets > 0.)
            allabels_p = torch.nonzero((sorted_p_and_n_targets > 0.)).flatten()
            labels_n = torch.nonzero((sorted_p_and_n_targets[:irrelevant_b_index] == 0)).flatten()

            fg_logits = all_buckets[labels_p]

            threshold_logit = torch.min(fg_logits) - delta_RS

            # Ignore those negative j that satisfy (L_{ij}=0 for all positive i), to accelerate the AP-loss computation.

            #Ranking loss computation with bucketed logits
            valid_labels_n = ((new_targets == 0) & (all_buckets >= threshold_logit))

            valid_bg_logits = all_buckets[valid_labels_n]

            bg_relations = (valid_bg_logits[:, None] - fg_logits[None, :]).cuda()

            fg_relations = (fg_logits[:, None] - fg_logits[None, :]).cuda()
            if delta_RS > 0:
                fg_relations = torch.clamp(fg_relations / (2 * delta_RS) + 0.5, min=0, max=1)
                bg_relations = torch.clamp(bg_relations / (2 * delta_RS) + 0.5, min=0, max=1)
            else:
                fg_relations = (fg_relations >= 0).float()
                bg_relations = (bg_relations >= 0).float()

            multiplication_bg = torch.mul(bg_relations, bucket_sizes_b)
            multiplication_fg = torch.mul(fg_relations, bucket_sizes_f)

            FP_num = torch.sum(multiplication_bg, axis=0)
            rank_pos = torch.sum(multiplication_fg, axis=0)


            rank = rank_pos + FP_num
            ranking_error = (FP_num / rank.float())

            #Compute sorting error with the same approach as in the original RankSort

            current_sorting_error = torch.sum(fg_relations.T * (1 - f_targets), axis=1) / rank_pos
    
            iou_relations = f_targets >= f_targets[:, None]

            target_sorted_order = fg_relations.T * iou_relations
            
            rank_pos_target = torch.sum(target_sorted_order, axis=1)

            target_sorting_error = torch.sum(target_sorted_order * (1 - f_targets), axis=1) / rank_pos_target
            sorting_error = current_sorting_error - target_sorting_error

            missorted_examples = fg_relations.t() * (~iou_relations)

            sorting_pmf_denom = torch.sum(missorted_examples, axis=1)
            FP_num[(FP_num == 0).nonzero()] = 1

            bucket_grads = (torch.sum(
                torch.mul(multiplication_bg, ranking_error * bucket_sizes_f.flatten()) / FP_num,
                axis=1) / bucket_sizes_b.flatten())

            duplication_bg = bucket_grads.repeat_interleave(bucket_sizes_b.flatten().type(torch.LongTensor).cuda())

            duplication_fg = ranking_error.repeat_interleave(
                bucket_sizes_f.flatten().type(torch.LongTensor).cuda())

            #Distribute grads into their original positions
            grad[p_and_n_indices[sorted_p_and_n_indices[:irrelevant_b_index][labels_n]]] = duplication_bg.cuda()
            grad[p_and_n_indices[sorted_p_and_n_indices[allabels_p]]] = -duplication_fg

            grad[p_and_n_indices[sorted_p_and_n_indices[allabels_p]]] -= sorting_error * (sorting_pmf_denom != 0.)
            x = sorting_error / sorting_pmf_denom

            # For positives, distribute error via sorting pmf (i.e. missorted_examples/sorting_pmf_denom)
            grad[p_and_n_indices[sorted_p_and_n_indices[allabels_p]]] += torch.sum(missorted_examples.t() * x, axis=1)

            #Normalize gradients by number of positives 
            grad /= fg_num

            metric = torch.sum(1 - ranking_error, dim=0) / fg_num

            ctx.save_for_backward(grad)
            return ranking_error.mean(), sorting_error.mean()
        #If there exists no relevant background logits, calculate only sorting loss
        else:
            ranking_error = torch.zeros(1).to(logits.device)
            
            fg_relations = (f_logits[:, None] - f_logits[None, :]).to(logits.device)
            if delta_RS > 0:
                fg_relations = torch.clamp(fg_relations / (2 * delta_RS) + 0.5, min=0, max=1)
            else:
                fg_relations = (fg_relations >= 0).float()

            bucket_sizes_f = torch.ones(len(f_logits)).to(logits.device)
            multiplication_fg = torch.mul(fg_relations, bucket_sizes_f)
            rank_pos = torch.sum(multiplication_fg, axis=0)
            
            current_sorting_error = torch.sum(fg_relations.T * (1 - f_targets), axis=1) / rank_pos
    
            iou_relations = f_targets >= f_targets[:, None]

            target_sorted_order = fg_relations.T * iou_relations

            
            rank_pos_target = torch.sum(target_sorted_order, axis=1)

            target_sorting_error = torch.sum(target_sorted_order * (1 - f_targets), axis=1) / rank_pos_target
            sorting_error = current_sorting_error - target_sorting_error

            missorted_examples = fg_relations.t() * (~iou_relations)

            sorting_pmf_denom = torch.sum(missorted_examples, axis=1)
            allabels_p = torch.nonzero((sorted_p_and_n_targets > 0.)).flatten()
            grad[p_and_n_indices[sorted_p_and_n_indices[allabels_p]]] -= sorting_error * (sorting_pmf_denom != 0.)
            x = sorting_error / sorting_pmf_denom

            # For positives, distribute error via sorting pmf (i.e. missorted_examples/sorting_pmf_denom)
            grad[p_and_n_indices[sorted_p_and_n_indices[allabels_p]]] += torch.sum(missorted_examples.t() * x, axis=1)

            #Normalize gradients by number of positives 
            grad /= fg_num

            metric = torch.sum(1 - ranking_error, dim=0) / fg_num

            ctx.save_for_backward(grad)
            return ranking_error.mean(), sorting_error.mean()

    @staticmethod
    def backward(ctx, out_grad1, out_grad2):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None, None

@LOSSES.register_module()
class RankSort(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, loss_weight=1,delta_RS=0.50, eps=1e-10): 

        classification_grads=torch.zeros(logits.shape).cuda()
        #Filter fg logits
        fg_labels = (targets > 0.)
        fg_logits = logits[fg_labels]
        fg_targets = targets[fg_labels]
        fg_num = len(fg_logits)
        if len(logits) == 0:
            ctx.save_for_backward(classification_grads)
            return torch.zeros(0).cuda(), torch.zeros(1).cuda()
        if fg_num == 0:
            ctx.save_for_backward(classification_grads)
            return torch.zeros(1).cuda(), torch.zeros(1).cuda()
        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta_RS
        relevant_bg_labels=((targets==0) & (logits>=threshold_logit))
        
        relevant_bg_logits = logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        sorting_error=torch.zeros(fg_num).cuda()
        ranking_error=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            # Difference Transforms (x_ij)
            fg_relations=fg_logits-fg_logits[ii] 
            bg_relations=relevant_bg_logits-fg_logits[ii]

            if delta_RS > 0:
                fg_relations=torch.clamp(fg_relations/(2*delta_RS)+0.5,min=0,max=1)
                bg_relations=torch.clamp(bg_relations/(2*delta_RS)+0.5,min=0,max=1)
            else:
                fg_relations = (fg_relations >= 0).float()
                bg_relations = (bg_relations >= 0).float()

            # Rank of ii among pos and false positive number (bg with larger scores)
            rank_pos=torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)

            # Rank of ii among all examples
            rank=rank_pos+FP_num
                            
            # Ranking error of example ii. target_ranking_error is always 0. (Eq. 7)
            ranking_error[ii]=FP_num/rank      

            # Current sorting error of example ii. (Eq. 7)
            current_sorting_error = torch.sum(fg_relations*(1-fg_targets))/rank_pos

            #Find examples in the target sorted order for example ii         
            iou_relations = (fg_targets >= fg_targets[ii])
            target_sorted_order = iou_relations * fg_relations

            #The rank of ii among positives in sorted order
            rank_pos_target = torch.sum(target_sorted_order)

            #Compute target sorting error. (Eq. 8)
            #Since target ranking error is 0, this is also total target error 
            target_sorting_error= torch.sum(target_sorted_order*(1-fg_targets))/rank_pos_target
            

            #Compute sorting error on example ii
            sorting_error[ii] = current_sorting_error - target_sorting_error
  
            #Identity Update for Ranking Error 
            if FP_num > eps:
                #For ii the update is the ranking error
                fg_grad[ii] -= ranking_error[ii]
                #For negatives, distribute error via ranking pmf (i.e. bg_relations/FP_num)
                relevant_bg_grad += (bg_relations*(ranking_error[ii]/FP_num))

            #Find the positives that are misranked (the cause of the error)
            #These are the ones with smaller IoU but larger logits
            missorted_examples = (~ iou_relations) * fg_relations

            #Denominotor of sorting pmf 
            sorting_pmf_denom = torch.sum(missorted_examples)

            #Identity Update for Sorting Error 
            if sorting_pmf_denom > eps:
                #For ii the update is the sorting error
                fg_grad[ii] -= sorting_error[ii]
                #For positives, distribute error via sorting pmf (i.e. missorted_examples/sorting_pmf_denom)
                fg_grad += (missorted_examples*(sorting_error[ii]/sorting_pmf_denom))

        #Normalize gradients by number of positives 
        classification_grads[fg_labels]= (fg_grad/fg_num)
        classification_grads[relevant_bg_labels]= (relevant_bg_grad/fg_num)

        ctx.save_for_backward(classification_grads)

        return ranking_error.mean(), sorting_error.mean()

    @staticmethod
    def backward(ctx, out_grad1, out_grad2):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None, None

class aLRPLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, regression_losses, delta=1., eps=1e-5): 
        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta

        #Get valid bg logits
        relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
        relevant_bg_logits=logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        rank=torch.zeros(fg_num).cuda()
        prec=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        max_prec=0                                           
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            #x_ij s as score differences with fgs
            fg_relations=fg_logits-fg_logits[ii] 
            #Apply piecewise linear function and determine relations with fgs
            fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
            #Discard i=j in the summation in rank_pos
            fg_relations[ii]=0

            #x_ij s as score differences with bgs
            bg_relations=relevant_bg_logits-fg_logits[ii]
            #Apply piecewise linear function and determine relations with bgs
            bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)

            #Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos=1+torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)
            #Store the total since it is normalizer also for aLRP Regression error
            rank[ii]=rank_pos+FP_num
                            
            #Compute precision for this example to compute classification loss 
            prec[ii]=rank_pos/rank[ii]                
            #For stability, set eps to a infinitesmall value (e.g. 1e-6), then compute grads
            if FP_num > eps:   
                fg_grad[ii] = -(torch.sum(fg_relations*regression_losses)+FP_num)/rank[ii]
                relevant_bg_grad += (bg_relations*(-fg_grad[ii]/FP_num))   
                    
        #aLRP with grad formulation fg gradient
        classification_grads[fg_labels]= fg_grad
        #aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels]= relevant_bg_grad 
 
        classification_grads /= (fg_num)
    
        cls_loss=1-prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss, rank, order

    @staticmethod
    def backward(ctx, out_grad1, out_grad2, out_grad3):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None, None, None
    
@LOSSES.register_module()
class APLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, delta=1.): 
        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets == 1)
        
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)
        if fg_num == 0:
            ctx.save_for_backward(classification_grads)
            return classification_grads.sum()
        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta

        #Get valid bg logits
        relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
        relevant_bg_logits=logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        rank=torch.zeros(fg_num).cuda()
        prec=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        max_prec=0                                           
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            #x_ij s as score differences with fgs
            fg_relations=fg_logits-fg_logits[ii] 
            #Apply piecewise linear function and determine relations with fgs
            fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
            #Discard i=j in the summation in rank_pos
            fg_relations[ii]=0

            #x_ij s as score differences with bgs
            bg_relations=relevant_bg_logits-fg_logits[ii]
            #Apply piecewise linear function and determine relations with bgs
            bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)

            #Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos=1+torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)
            #Store the total since it is normalizer also for aLRP Regression error
            rank[ii]=rank_pos+FP_num
                            
            #Compute precision for this example 
            current_prec=rank_pos/rank[ii]
            
            #Compute interpolated AP and store gradients for relevant bg examples
            if (max_prec<=current_prec):
                max_prec=current_prec
                relevant_bg_grad += (bg_relations/rank[ii])
            else:
                relevant_bg_grad += (bg_relations/rank[ii])*(((1-max_prec)/(1-current_prec)))
            
            #Store fg gradients
            fg_grad[ii]=-(1-max_prec)
            prec[ii]=max_prec 

        #aLRP with grad formulation fg gradient
        classification_grads[fg_labels]= fg_grad
        #aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels]= relevant_bg_grad 
 
        classification_grads /= fg_num
    
        cls_loss=1-prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss

    @staticmethod
    def backward(ctx, out_grad1):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None
