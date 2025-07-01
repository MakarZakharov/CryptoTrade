"""
Model Approval Workflow

Manages the approval process for model deployment and updates.
"""

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging


class ApprovalStatus(Enum):
    """Model approval status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class ModelApprovalRequest:
    """Model approval request data structure"""
    request_id: str
    model_name: str
    model_version: str
    submitter: str
    submission_date: datetime
    model_metrics: Dict[str, float]
    test_results: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    status: ApprovalStatus = ApprovalStatus.PENDING
    reviewer: Optional[str] = None
    review_date: Optional[datetime] = None
    comments: List[str] = None
    
    def __post_init__(self):
        if self.comments is None:
            self.comments = []


class ModelApprovalWorkflow:
    """Manages model approval workflow"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize approval workflow"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.pending_requests: Dict[str, ModelApprovalRequest] = {}
        self.completed_requests: Dict[str, ModelApprovalRequest] = {}
        
        # Approval thresholds
        self.thresholds = {
            'min_accuracy': 0.6,
            'max_drawdown': 0.2,
            'min_sharpe_ratio': 1.0,
            'max_var_95': 0.05
        }
        self.thresholds.update(self.config.get('thresholds', {}))
        
    def submit_approval_request(
        self,
        model_name: str,
        model_version: str,
        submitter: str,
        model_metrics: Dict[str, float],
        test_results: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> str:
        """
        Submit model for approval
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            submitter: Person submitting the request
            model_metrics: Performance metrics
            test_results: Test results and validation
            risk_assessment: Risk assessment results
            
        Returns:
            Request ID
        """
        request_id = str(uuid.uuid4())
        
        request = ModelApprovalRequest(
            request_id=request_id,
            model_name=model_name,
            model_version=model_version,
            submitter=submitter,
            submission_date=datetime.now(),
            model_metrics=model_metrics,
            test_results=test_results,
            risk_assessment=risk_assessment
        )
        
        self.pending_requests[request_id] = request
        
        # Perform automated checks
        self._perform_automated_checks(request)
        
        self.logger.info(f"Model approval request submitted: {request_id}")
        return request_id
        
    def _perform_automated_checks(self, request: ModelApprovalRequest) -> None:
        """Perform automated approval checks"""
        metrics = request.model_metrics
        issues = []
        
        # Check accuracy threshold
        if metrics.get('accuracy', 0) < self.thresholds['min_accuracy']:
            issues.append(f"Accuracy {metrics.get('accuracy', 0):.3f} below threshold {self.thresholds['min_accuracy']}")
            
        # Check drawdown
        if metrics.get('max_drawdown', 1) > self.thresholds['max_drawdown']:
            issues.append(f"Max drawdown {metrics.get('max_drawdown', 1):.3f} exceeds threshold {self.thresholds['max_drawdown']}")
            
        # Check Sharpe ratio
        if metrics.get('sharpe_ratio', 0) < self.thresholds['min_sharpe_ratio']:
            issues.append(f"Sharpe ratio {metrics.get('sharpe_ratio', 0):.3f} below threshold {self.thresholds['min_sharpe_ratio']}")
            
        # Check VaR
        if metrics.get('var_95', 1) > self.thresholds['max_var_95']:
            issues.append(f"VaR 95% {metrics.get('var_95', 1):.3f} exceeds threshold {self.thresholds['max_var_95']}")
            
        if issues:
            request.status = ApprovalStatus.REQUIRES_REVIEW
            request.comments.extend(issues)
        else:
            # Auto-approve if all checks pass
            self._approve_request(request.request_id, "system", "Automated approval - all checks passed")
            
    def review_request(
        self,
        request_id: str,
        reviewer: str,
        decision: ApprovalStatus,
        comments: str
    ) -> bool:
        """
        Review a pending approval request
        
        Args:
            request_id: Request ID to review
            reviewer: Person reviewing the request
            decision: Approval decision
            comments: Review comments
            
        Returns:
            True if review was successful
        """
        if request_id not in self.pending_requests:
            self.logger.error(f"Request {request_id} not found")
            return False
            
        request = self.pending_requests[request_id]
        request.reviewer = reviewer
        request.review_date = datetime.now()
        request.status = decision
        request.comments.append(f"Review by {reviewer}: {comments}")
        
        if decision in [ApprovalStatus.APPROVED, ApprovalStatus.REJECTED]:
            # Move to completed requests
            self.completed_requests[request_id] = request
            del self.pending_requests[request_id]
            
        self.logger.info(f"Request {request_id} reviewed by {reviewer}: {decision.value}")
        return True
        
    def _approve_request(self, request_id: str, reviewer: str, comments: str) -> None:
        """Approve a request"""
        if request_id in self.pending_requests:
            request = self.pending_requests[request_id]
            request.reviewer = reviewer
            request.review_date = datetime.now()
            request.status = ApprovalStatus.APPROVED
            request.comments.append(comments)
            
            self.completed_requests[request_id] = request
            del self.pending_requests[request_id]
            
    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests"""
        return [asdict(request) for request in self.pending_requests.values()]
        
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific request"""
        if request_id in self.pending_requests:
            return asdict(self.pending_requests[request_id])
        elif request_id in self.completed_requests:
            return asdict(self.completed_requests[request_id])
        return None
        
    def get_approval_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Get approval history for a model"""
        history = []
        
        # Check completed requests
        for request in self.completed_requests.values():
            if request.model_name == model_name:
                history.append(asdict(request))
                
        # Check pending requests
        for request in self.pending_requests.values():
            if request.model_name == model_name:
                history.append(asdict(request))
                
        return sorted(history, key=lambda x: x['submission_date'], reverse=True)