
module combined_layers
  use structured_network
  implicit none

  type, extends(layer) :: additive_layer
    class(network), allocatable :: internals(:)
  contains
    procedure :: train_forward => train_forward_additive_layer
    procedure :: train_backward => backpropagate_additive_layer
    procedure :: run => run_additive_layer
    procedure :: random_init => additive_random_init
    procedure :: f_init1 => additive_layer_fixed_init
  end type additive_layer

  type, extends(layer) :: self_attention_head
    real, allocatable :: embed_K(:,:), embed_Q(:,:), embed_V(:,:), KQT(:,:,:), KQT_derivatives(:,:,:)
  contains
    procedure :: train_forward => train_forward_self_attention
    procedure :: train_backward => backpropagate_self_attention
    procedure :: run => run_self_attention
    procedure :: random_init => self_attention_randomize
    procedure :: f_init1 => self_attention_fixed_init
  end type self_attention_head

  interface self_attention_head
    module procedure self_attention_from_params
    module procedure self_attention_from_activation_and_params
  end interface self_attention_head

contains

  pure subroutine additive_layer_fixed_init(this, value)
    class(additive_layer), intent(inout) :: this
    real, intent(in) :: value
    integer :: i,j
    do concurrent (i=1:size(this%internals))
      do concurrent (j=1:size(this%internals(i)%layers))
        call this%internals(i)%layers(j)%f_init1(value)
      end do
    end do
  end subroutine additive_layer_fixed_init

  function additive_layer_from_networks(net1, net2) result (out)
    type(additive_layer) :: out
    type(network), target :: net1, net2

    !assert net1%outputs == net2%outputs
    out%outputs = net1%outputs

    allocate(out%internals, source=[net1,net2])
    out%inputs=net1%layers(1)%inputs + net2%layers(1)%inputs
  end function additive_layer_from_networks

  subroutine additive_random_init(this)
    class(additive_layer), intent(inout) :: this
    integer :: i
    do i=1,size(this%internals)
      call this%internals(i)%initiate()
    end do
  end subroutine additive_random_init

  pure function run_additive_layer(this, signals) result (out)
    class(additive_layer), intent(in) :: this
    real, intent(in) :: signals(:,:) ! 1st dim this%inputs long
    real :: out(this%outputs, size(signals,2))
    real :: intermediate(this%outputs, size(signals,2),size(this%internals))
    integer :: i, delimiters(size(this%internals)+1)

    do i=1,size(this%internals)+1
      delimiters(i+1) = delimiters(i)+this%internals(i)%layers(1)%inputs
    end do

    do concurrent (i=1:size(this%internals))
      intermediate(:,:,i) = this%internals(i)%run(   &
        signals( delimiters(i):delimiters(i+1) , : ) &
      )
    end do

    out = sum(intermediate, 3)
    if (allocated(this%func)) out=this%func%activate(out)
  end function run_additive_layer

  function train_forward_additive_layer(this, signals) result (out)
    class(additive_layer), intent(inout) :: this
    real, intent(in) :: signals(:, :) ! 1st dim needs to be this%inputs long!
    real :: out(this%outputs,size(signals,2)), intermediate(this%outputs,size(signals,2),size(this%internals))
    integer :: i, delimiters(size(this%internals)+1)

    if (allocated(this%signals)) deallocate(this%signals)
    allocate(this%signals, source=signals)

    do i=1,size(this%internals)+1
      delimiters(i+1) = delimiters(i)+this%internals(i)%layers(1)%inputs
    end do

    do i=1,size(this%internals)
      intermediate(:,:,i)=this%internals(i)%forward( &
        signals( delimiters(i):delimiters(i+1) , : )  &
        )
    end do

    out = sum(intermediate,3)
    if (allocated(this%func)) then
      if (allocated(this%derivatives)) deallocate(this%derivatives)
      allocate(this%derivatives, mold=out)
      block
        integer :: i
        do concurrent(i=1:size(this%signals,2))
          this%derivatives(:,i)=this%func%derivative(out(:,i))
          out(:,i) = this%func%activate(out(:,i))
        end do
      end block
    end if
  end function train_forward_additive_layer

  function backpropagate_additive_layer(this, error, alpha) result (out)
    class(additive_layer), intent(inout) :: this
    real, intent(in) :: error(:,:), alpha
    real :: out(this%inputs, size(error,2)), intermediate(this%outputs, size(error,2))
    integer :: i, delimiters(size(this%internals)+1)

    intermediate = error
    if (allocated(this%func)) intermediate = intermediate * this%derivatives

    do i=1,size(this%internals)+1
      delimiters(i+1) = delimiters(i)+this%internals(i)%layers(1)%inputs
    end do

    do i=1,size(this%internals)
      block
        real, allocatable :: rorre(:,:)
        allocate(rorre, source=intermediate)
        call this%internals(i)%backward(rorre,alpha)
        out(delimiters(i):delimiters(i+1), :) = rorre
        deallocate(rorre)
      end block
    end do
  end function backpropagate_additive_layer

  function self_attention_from_params(channels, embedded) result (self_attention)
    type(self_attention_head) :: self_attention
    integer, intent(in) :: channels, embedded
    self_attention%outputs = embedded
    self_attention%inputs = channels
    allocate(self_attention%embed_K(embedded, channels))
    allocate(self_attention%embed_Q(embedded, channels))
    allocate(self_attention%embed_V(embedded, channels))
  end function self_attention_from_params

  function self_attention_from_activation_and_params(func, channels, embedded) result (self_attention)
    type(self_attention_head) :: self_attention
    class(Activation), intent(in) :: func
    integer, intent(in) :: channels, embedded
    allocate(self_attention%func, source=func)
    self_attention%outputs = embedded
    self_attention%inputs = channels
    allocate(self_attention%embed_K(embedded, channels))
    allocate(self_attention%embed_Q(embedded, channels))
    allocate(self_attention%embed_V(embedded, channels))
  end function self_attention_from_activation_and_params

  subroutine self_attention_randomize(this)
    class(self_attention_head), intent(inout) :: this
    call random_number(this%embed_K)
    call random_number(this%embed_Q)
    call random_number(this%embed_V)
  end subroutine self_attention_randomize

  pure subroutine self_attention_fixed_init(this, value)
    class(self_attention_head), intent(inout) :: this
    real, intent(in) :: value
    this%embed_K=value
    this%embed_Q=value
    this%embed_V=value
  end subroutine self_attention_fixed_init

  pure function run_self_attention(this, signals) result (out)
    class(self_attention_head), intent(in) :: this
    real, intent(in) :: signals(:,:)
    real :: out(this%outputs, size(signals,2)), KQT(size(this%embed_K,1), size(this%embed_Q, 1))
    integer :: i
    do concurrent (i=1:size(signals,2))
      KQT = matmul(                           &
        reshape(                              &
          matmul(this%embed_K, signals(:,i)), &
          [size(this%embed_K,1),1]            &
        ),                                    &
        reshape(                              &
          matmul(this%embed_Q, signals(:,i)), &
          [1,size(this%embed_Q,1)]            &
        )                                     &
      )
      call soft_mark(KQT)
      out(:,i) = matmul(KQT, matmul(this%embed_V, signals(:,i)))
    end do
  end function run_self_attention

  function train_forward_self_attention(this, signals) result (out)
    class(self_attention_head), intent(inout) :: this
    real, intent(in) :: signals(:,:)
    real :: out(this%outputs, size(signals,2))
    integer :: i

    if (allocated(this%signals)) deallocate(this%signals)
    allocate(this%signals, source=signals)
    if (allocated(this%KQT)) deallocate(this%KQT)
    allocate(this%KQT(size(this%embed_K,1),size(this%embed_Q,1),size(this%embed_V,2)), source=0.)
    if (allocated(this%KQT_derivatives)) deallocate(this%KQT_derivatives)
    allocate(this%KQT_derivatives(size(this%embed_K,1),size(this%embed_Q,1),size(this%embed_V,2)),source=0.)

    do concurrent(i=1:this%inputs)
      this%KQT(:,:,i) = matmul(                              &
        reshape(this%embed_K(:,i),[size(this%embed_K,1),1]), &
        reshape(this%embed_Q(:,i),[1,size(this%embed_Q,1)])  &
      )
      block !soft_mask
        real :: exponentiated(size(this%KQT,1))
        integer :: row

        exponentiated = 0

        do concurrent(row=1:size(this%KQT,1))
        ! this might be slower on GPU?
          !sft_arr(row, row+1:) = 0
          exponentiated(1:row)=exp(this%KQT(row,1:row,i))
        ! Might be better to exp everything and set exponentiated(row+1:) = 0
          if (any(exponentiated(1:row)>0)) then
            this%KQT(row,1:row,i) = exponentiated(1:row) / sum(exponentiated(1:row))    / row
            this%KQT_derivatives(row,1:row,i) = (sum(exponentiated(1:row))-exponentiated(1:row))* &
                                    exponentiated(1:row) / sum(exponentiated(1:row))**2 / row
          end if
        end do
      end block
    end do

    block !actual output calculation
      type(one_hot) :: decoder
      integer :: channels(size(signals,2))

      decoder = one_hot(size(this%embed_V, 2))

      !assume signals is range of one-hots
      channels = decoder%decode(signals)

      do concurrent (i=1:size(signals,2))
        out(:,i) = matmul(            &
          this%KQT(:,:,channels(i)),  &
          this%embed_V(:,channels(i)) &
        )
      end do
    end block

    if (allocated(this%func)) then
      if (allocated(this%derivatives)) deallocate(this%derivatives)
      allocate(this%derivatives(this%outputs, size(signals,2)))
      do concurrent (i=1:size(signals,2))
        this%derivatives(:,i)=this%func%derivative(out(:,i))
        out(:,i:i) = this%func%activate(out(:,i:i))
      end do
    end if

  end function train_forward_self_attention

  function backpropagate_self_attention(this, error, alpha) result (out)
    class(self_attention_head), intent(inout) :: this
    real, intent(in) :: error(:,:), alpha
    type(one_hot)::decoder
    real :: errorQ(size(this%embed_Q,1),size(this%embed_Q,2))
    real :: errorK(size(this%embed_Q,1),size(this%embed_Q,2))
    real :: errorV(size(this%embed_Q,1),size(this%embed_Q,2))
    real :: out(this%inputs,size(error,2)), derivative_times_error(this%outputs, size(error,2))
    integer :: i, channels(size(error,2))
    real :: KQT_err(this%outputs, size(this%embed_V,1))

    errorK = 0
    errorQ = 0
    errorV = 0
    channels = decoder%decode(this%signals)

    if (allocated(this%func))then
      derivative_times_error = error*this%derivatives
      deallocate(this%derivatives)
    else
      derivative_times_error = error
    end if

    do concurrent (i=1:size(error,2))
      KQT_err = matmul(                                    &
        derivative_times_error(:,i:i),                     &
        transpose(this%embed_V(:,channels(i):channels(i))) &
      )
      KQT_err = KQT_err * this%KQT_derivatives(:,:,channels(i))

      ! errorK = KQT_err x embed_Q  = (K x Q^T)' x Q = K'
      errorK(:,channels(i)) = errorK(:,channels(i)) + matmul(KQT_err, this%embed_Q(:,channels(i)))
      ! errorQ = (t(embed_K) x KQT_err)^T = (K^T x (KxQ^T)')^T = Q^T^T=Q
      errorQ(:,channels(i)) = errorQ(:,channels(i)) + matmul(transpose(KQT_err), this%embed_K(:,channels(i)))
      ! pretty much useless, but have to provide out.
      out(:,i) = matmul(transpose(this%KQT(:,:,channels(i))), derivative_times_error(:,i))
      errorV(:,channels(i)) = errorV(:,channels(i)) + out(:,i)
      !could call backpropagate on Q & K here if they were layers!
    end do
    deallocate(this%KQT_derivatives)
    deallocate(this%KQT)
    !could average by row?
    errorQ = errorQ/size(error,2)
    errorK = errorK/size(error,2)
    errorV = errorV/size(error,2)

    this%embed_Q = this%embed_Q - errorQ*alpha
    this%embed_K = this%embed_K - errorK*alpha
    this%embed_V = this%embed_V - errorV*alpha

  end function backpropagate_self_attention

! internal / shortcut function!
  function soft_mask(sft_arr) result (derivatives)
    real, intent(inout) :: sft_arr(:,:)
    real :: derivatives(size(sft_arr,1), size(sft_arr,2))
    real :: exponentiated(size(sft_arr,1))
    integer :: row

    derivatives = 0
    exponentiated = 0

    do concurrent(row=1:size(sft_arr,1))
    ! this might be slower on GPU?
      !sft_arr(row, row+1:) = 0
      exponentiated(1:row)=exp(sft_arr(row,1:row))
    ! Might be better to exp everything and set exponentiated(row+1:) = 0
      sft_arr(row,1:row) = exponentiated(1:row) / sum(exponentiated(1:row))
      !derivative(row,1:row) = exponentiated / sum(exponentiated) - exponentiated**2/sum(exponentiated)**2
      ! sooooo:
      derivatives(row,1:row) = sft_arr(row, 1:row) - sft_arr(row, 1:row)**2
    end do
  end function soft_mask

  pure subroutine soft_mark(sft_arr)
    real, intent(inout) :: sft_arr(:,:)
    real :: exponentiated(size(sft_arr,1))
    integer :: row
    exponentiated=0
    do concurrent(row = 1:size(sft_arr, 1))
      !sft_arr(row, row+1:) = 0
      exponentiated(1:row)=exp(sft_arr(row,1:row))
      sft_arr(row, :)=exponentiated / sum(exponentiated)
    end do
  end subroutine soft_mark

end module combined_layers