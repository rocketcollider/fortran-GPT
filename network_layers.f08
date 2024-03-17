
module network_layers
  use network_utils
  implicit none

!========================== abstract layer ==========================

  type, abstract :: layer
    integer :: inputs, outputs
    class(Activation), allocatable :: func
    real, allocatable :: weights(:,:), signals(:,:), derivatives(:,:)
  contains
    procedure :: set_layout => set_weights_size
    procedure :: random_weights => set_weights_random
    procedure(layer_pass_signals), deferred :: run
    procedure(layer_train_forward), deferred :: train_forward
    procedure(layer_calc_gradient), deferred :: calc_grad
    procedure(layer_train_backward), deferred :: train_backward
  end type layer

!========================== interface declaration ==========================

  abstract interface
    function layer_pass_signals(this,signals) result (out)
    import layer
      class(layer), intent(in) :: this
      real, intent(in) :: signals(:,:) ! 1st dim this%inputs long
      real :: out(this%outputs, size(signals,2))
    end function layer_pass_signals

    function layer_train_forward(this,signals) result (out)
    import layer
      class(layer), intent(inout) :: this
      real, intent(in) :: signals(:,:) ! 1st dim this%inputs long
      real :: out(this%outputs, size(signals,2))
    end function layer_train_forward

    function layer_train_backward(this,error, alpha) result (out)
      import layer
      class(layer), intent(inout) :: this
      real, intent(in) :: alpha, error(:,:) ! 1st dim this%outputs long
      real :: out(this%inputs, size(error,2))
    end function layer_train_backward

    function layer_calc_gradient(this, error, corrections) result (out)
      import layer
      class(layer), intent(in) :: this
      real, intent(in) :: error(:,:) ! 1st dim this%outputs long
      real, intent(out) :: corrections(size(this%weights,1), size(this%weights,2))
      real :: out(this%inputs, size(error,2))
    end function layer_calc_gradient
  end interface

!========================== useful definitions ==========================

  type, extends(layer) :: bias_layer
  contains
    procedure :: set_layout => set_biased_layout
    procedure :: train_forward => train_forward_biased_layer
    procedure :: train_backward => train_backward_biased_layer
    procedure :: calc_grad => gradient_biased_layer
    procedure :: run => interpret_biased_layer
  end type bias_layer

  interface bias_layer
    module procedure bias_layer_from_activation
    module procedure bias_layer_from_activation_and_size
  end interface bias_layer

contains

  subroutine set_biased_layout(this, inputs, outputs)
    class(bias_layer), intent(inout) :: this
    integer, intent(in) :: inputs, outputs
    this%inputs = inputs
    this%outputs = outputs
    if (allocated(this%weights)) deallocate(this%weights)
    allocate(this%weights(outputs,inputs+1))
  end subroutine set_biased_layout

  subroutine set_weights_size(this, inputs, outputs)
    class(layer), intent(inout) :: this ! HAS to be inout, or other params are lost !!
    integer, intent(in) :: inputs, outputs
    this%inputs = inputs
    this%outputs = outputs
    if (allocated(this%weights)) deallocate(this%weights)
    allocate(this%weights(outputs,inputs))
  end subroutine set_weights_size

  function bias_layer_from_activation(func) result (out)
    class(Activation), intent(in) :: func
    type(bias_layer) :: out
    allocate(out%func, source=func)
  end function bias_layer_from_activation

  function bias_layer_from_activation_and_size(func, inputs, outputs) result (out)
    class(Activation), intent(in) :: func
    type(bias_layer) :: out
    integer, intent(in) :: inputs, outputs
    call out%set_layout(inputs, outputs)
    allocate(out%func, source=func)
  end function bias_layer_from_activation_and_size

  subroutine set_weights_random(this)
    class(layer) :: this
    call random_number(this%weights)
  end subroutine set_weights_random

  function interpret_biased_layer(this, signals) result (out)
    class(bias_layer), intent(in) :: this
    real, intent(in) :: signals(:,:)
    real :: out(this%outputs,size(signals,2)), wrapper(this%inputs+1,size(signals,2))
    wrapper(this%inputs+1,:)=1
    wrapper(: this%inputs,:)=signals
    out = this%func%activate(matmul(this%weights, wrapper))
  end function interpret_biased_layer

  function train_forward_biased_layer(this, signals) result (out)
    class(bias_layer), intent(inout) :: this
    real, intent(in) :: signals(:, :) ! 1st dim needs to be this%inputs long!
    real :: out(this%outputs,size(signals,2))

    if (allocated(this%signals)) deallocate(this%signals)
    allocate(this%signals(this%inputs+1,size(signals,2)))
    this%signals(this%inputs+1, :)=1
    this%signals(: this%inputs, :)=signals

    out = matmul(   &
      this%weights, &
      this%signals  &
    )
    if (allocated(this%derivatives)) deallocate(this%derivatives)
    allocate(this%derivatives, mold=out)!source = this%func%derivative(out))
    block
      integer :: i
      do concurrent(i=1:size(signals,2))
        this%derivatives(:,i)=this%func%derivative(out(:,i))
        out(:,i) = this%func%activate(out(:,i))
      end do
    end block
  end function train_forward_biased_layer

  function gradient_biased_layer(this, error, corrections) result (prev_error)
    class(bias_layer), intent(in) :: this
    real, intent(in) :: error(:,:) ! 1st dim needs to be this%outputs long!
    real :: prev_error(this%inputs, size(error,2))
    real, intent(out) :: corrections(size(this%weights,1), size(this%weights,2))

    if (.not. allocated(this%signals) .or. .not. allocated(this%derivatives)) then
      print *, "THIS NEEDS SIGNALS AND DERIVATIVES! RUN TRAIN_FORWARD FIRST!!!"
      stop
    end if

    prev_error = matmul(                         &
      transpose(                                 &
        this%weights(:,1:size(this%weights,2)-1) &
      ),                                         &
      this%derivatives * error                   &
    )

    corrections = matmul( &
      error,              &
      transpose(          &
        this%signals      &
      )                   &
    ) / size(error,2)

  end function gradient_biased_layer

  function train_backward_biased_layer(this, error, alpha) result (prev_error)
    class(bias_layer), intent(inout)::this
    real,intent(in) :: error(:,:), alpha
    real :: prev_error(this%inputs, size(error,2))!,corrections(this%outputs, this%inputs+1)

    prev_error = basic_SGD(this,error,alpha)
    !prev_error=this%calc_grad(error, corrections)
    !this%weights = this%weights - (corrections * alpha)
    deallocate(this%signals)
    deallocate(this%derivatives)
  end function train_backward_biased_layer

  function basic_SGD(this, error, alpha) result (prev_error)
    class(layer), intent(inout)::this
    real, intent(in) :: error(:,:), alpha ! error needs to be this%outputs long in 1st dim
    real :: corrections(size(this%weights,1),size(this%weights,2)), prev_error(this%inputs, size(error,2))

    prev_error=this%calc_grad(error, corrections)
    this%weights = this%weights - corrections * alpha
  end function basic_SGD

end module network_layers