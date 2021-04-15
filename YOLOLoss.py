import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    def __init__(self,mBatch,mB=2,mC=20,lamda_coord=5,lamda_noobj=0.5):
        super(YOLOLoss,self).__init__()
        self.B = mB
        self.C = mC
        self.Batch_Size = mBatch
        self.lamda_coord = lamda_coord
        self.lamda_noobj = lamda_noobj
        
    def compute_iou(self):
        return None
        
    def forward(self,pred_tsr,target_tsr):
        # Target Tsr is a (B,7,7,30) Tensor
        # Pred Tsr is a (B,7*7*30) Tensor
        # Step 1: Reshape Two Input Tensor to (B,7*7,30) Size
        n_element = self.B * 5 + self.C
        target_tsr = target_tsr.view(self.Batch_Size,-1,n_element)
        pred_tsr = pred_tsr.view(self.Batch_Size,-1,n_element)
        
        # Step 2: Make Contain Objects & Not Contain Objects Mask
        # 1) Contain Objects
        contain_obj_mask = target_tsr[:,:,4] > 0
        contain_obj_mask = contain_obj_mask.unsqueeze(-1).expand_as(target_tsr)
        contain_obj_target = target_tsr[contain_obj_mask].view(-1,n_element)
        contain_obj_pred = pred_tsr[contain_obj_mask].view(-1,n_element)
        Class_Pred = contain_obj_pred[:,10:]
        BBox_Pred = contain_obj_pred[:,:10].contiguous().view(-1,5)
        Class_Target = contain_obj_target[:,10:]
        BBox_Target = contain_obj_target[:,:10].contiguous().view(-1,5)
        
        # 2) No Objects
        no_obj_mask = target_tsr[:,:,4] == 0
        no_obj_mask = no_obj_mask.unsqueeze(-1).expand_as(target_tsr)
        no_obj_target = target_tsr[no_obj_mask].view(-1,n_element)
        no_obj_pred = pred_tsr[no_obj_mask].view(-1,n_element)
        
        # Step 3: Calculate Not Contain Object Loss
        # Tips: Only Calculate The Bias Of Confidence, Other Params Are Not Considered.
        Confidence_Mask = torch.zeros(size=no_obj_pred.size()).bool()
        Confidence_Mask[:,4] = True
        Confidence_Mask[:,9] = True
        No_Obj_Pred_Confidence = no_obj_pred[Confidence_Mask]
        No_Obj_Target_Confidence = no_obj_target[Confidence_Mask]
        No_Obj_Loss = self.lamda_noobj * F.mse_loss(No_Obj_Pred_Confidence,No_Obj_Target_Confidence,reduction='mean')
        # Step 4: Calculate Contain Object Loss
        # Tips Contain Object Loss Is Combined With Four Parts
        # [1) Center Point Loss 2) W & H Loss] 3) Confidence Loss 4) Classification Loss
        Contain_Object_Loc_Loss_xy = self.lamda_coord * F.mse_loss(BBox_Pred[:,:2],BBox_Target[:,:2],reduction='mean')
        Contain_Object_Loc_Loss_wh = self.lamda_coord * F.mse_loss(BBox_Pred[:,2:4],BBox_Target[:,2:4],reduction='mean')
        Contain_Object_Loc_Loss_c = F.mse_loss(BBox_Pred[:,-1],BBox_Target[:,-1],reduction='mean')
        Contain_Object_Loc_Loss = Contain_Object_Loc_Loss_c + Contain_Object_Loc_Loss_wh + Contain_Object_Loc_Loss_xy
        # Classification Loss
        Contain_Object_Cls_Loss = F.mse_loss(Class_Pred,Class_Target,reduction='mean')
        return Contain_Object_Cls_Loss + Contain_Object_Loc_Loss + No_Obj_Loss