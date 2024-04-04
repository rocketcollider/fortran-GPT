
module network_encoding
  implicit none

  type, abstract :: encoding
    integer :: outputs
  contains
    generic :: encode => eval0, eval1, eval2, eval3
    procedure(eval_encoding), deferred :: eval0
    procedure :: eval1 => generic_encode1
    procedure :: eval2 => generic_encode2
    procedure :: eval3 => generic_encode3
    generic :: decode => decode0, decode1, decode2, decode3
    procedure(reverse_encode), deferred :: decode0
    procedure :: decode1 => generic_decode1
    procedure :: decode2 => generic_decode2
    procedure :: decode3 => generic_decode3
    procedure :: train => untrainable
  end type encoding

  abstract interface
    pure function eval_encoding(this, data) result (out)
      import encoding
      class(encoding), intent(in) :: this
      integer, intent(in) :: data
      real :: out(this%outputs)
    end function eval_encoding

    pure function reverse_encode(this, logits) result (out)
      import encoding
      class(encoding), intent(in) :: this
      real, intent(in) :: logits(this%outputs)
      integer :: out
    end function reverse_encode
  end interface

  type, extends(encoding) :: one_hot
  contains
    procedure :: eval0 => one_hot_eval
    procedure :: decode0 => one_cold0
    procedure :: decode1 => one_cold1
    procedure :: decode2 => one_cold2
    procedure :: decode3 => one_cold3
  end type one_hot

  type, extends(encoding) :: embedding
    real, allocatable :: weights(:,:)
  contains
    procedure :: eval0 => embedding_eval
    procedure :: decode0 => embedding_reverse
    procedure :: train => train_embedding
  end type embedding
contains

!========= generic stuff =============

  subroutine train_embedding(this, error, alpha)
    class(embedding), intent(inout) :: this
    real, intent(in) :: error(:,:), alpha
    this%weights = this%weights - error*alpha
  end subroutine train_embedding

  pure function embedding_eval(this, data) result (out)
    class(embedding), intent(in) :: this
    integer, intent(in) :: data
    real :: out(this%outputs)
    out = this%weights(:,data)
  end function embedding_eval

  pure function embedding_reverse(this, logits) result (out)
    class(embedding), intent(in) :: this
    real, intent(in) :: logits(this%outputs)
    integer :: i, out
    real :: projections(size(this%weights,1))
    do concurrent (i=1:this%outputs)
      projections(i) = dot_product(this%weights(i,:), logits)
    end do
    out = maxloc(projections,1)
  end function embedding_reverse

  pure function generic_encode1(this, data) result(out)
    class(encoding), intent(in) :: this
    integer, intent(in):: data(:)
    integer :: i
    real :: out(this%outputs, size(data,1))
    do concurrent (i=1:size(data,1))
      out(:,i) = this%eval0(data(i))
    end do
  end function generic_encode1

  pure function generic_encode2(this, data) result(out)
    class(encoding), intent(in) :: this
    integer, intent(in):: data(:,:)
    integer :: i,j
    real :: out(this%outputs, size(data,1),size(data,2))
    do concurrent (i=1:size(data,1))
      do concurrent (j=1:size(data,2))
        out(:,i,j) = this%eval0(data(i,j))
      end do
    end do
  end function generic_encode2

  pure function generic_encode3(this, data) result(out)
    class(encoding), intent(in) :: this
    integer, intent(in):: data(:,:,:)
    integer :: i,j,k
    real :: out(this%outputs, size(data,1),size(data,2),size(data,3))
    do concurrent (i=1:size(data,1))
      do concurrent (j=1:size(data,2))
        do concurrent (k=1:size(data,3))
          out(:,i,j,k) = this%eval0(data(i,j,k))
        end do
      end do
    end do
  end function generic_encode3

  pure function generic_decode1(this, logits) result(out)
    class(encoding), intent(in) :: this
    real, intent(in) :: logits(:,:)
    integer :: out(size(logits,2))
    integer :: i
    do concurrent (i=1:size(logits,1))
      out(i) = this%decode0(logits(:,i))
    end do
  end function generic_decode1

  pure function generic_decode2(this, logits) result(out)
    class(encoding), intent(in) :: this
    real, intent(in) :: logits(:,:,:)
    integer :: out(size(logits,2),size(logits,3))
    integer :: i,j
    do concurrent (i=1:size(logits,1))
      do concurrent (j=1:size(logits,2))
        out(i,j) = this%decode0(logits(:,i,j))
      end do
    end do
  end function generic_decode2

  pure function generic_decode3(this, logits) result(out)
    class(encoding), intent(in) :: this
    real, intent(in) :: logits(:,:,:,:)
    integer :: out(size(logits,2), size(logits,3),size(logits,4))
    integer :: i,j,k
    do concurrent (i=1:size(logits,1))
      do concurrent (j=1:size(logits,2))
        do concurrent (k=1:size(logits,3))
          out(i,j,k) = this%decode0(logits(:,i,j,k))
        end do
      end do
    end do
  end function generic_decode3

  subroutine untrainable(this, error, alpha)
    class(encoding), intent(inout) :: this
    real, intent(in) :: error(:,:), alpha

  end subroutine untrainable

!========= generic stops ===========

  pure function one_hot_eval(this, data) result (out)
    class(one_hot), intent(in) :: this
    integer, intent(in) :: data
    real :: out(this%outputs)
    out = 0
    out(data) = 1
  end function one_hot_eval

  pure function one_cold0(this, logits) result (out)
    class(one_hot), intent(in) :: this
    real, intent(in) :: logits(this%outputs)
    integer :: out
    out = maxloc(logits,1)
  end function

  pure function one_cold1(this, logits) result (out)
    class(one_hot), intent(in) :: this
    real, intent(in) :: logits(:,:)
    integer :: out(size(logits,2))
    out = maxloc(logits,dim=1)
  end function

  pure function one_cold2(this, logits) result (out)
    class(one_hot), intent(in) :: this
    real, intent(in) :: logits(:,:,:)
    integer :: out(size(logits,2),size(logits,3))
    out = maxloc(logits,1)
  end function

  pure function one_cold3(this, logits) result (out)
    class(one_hot), intent(in) :: this
    real, intent(in) :: logits(:,:,:,:)
    integer :: out(size(logits,2),size(logits,3),size(logits,4))
    out = maxloc(logits,1)
  end function

end module network_encoding