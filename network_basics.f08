
module network_basics
  use network_utils
  implicit none

  type evaluation
    real,allocatable :: vals(:,:)
  end type evaluation

  interface layer
    module procedure layer_from_wrapper, new_layer
  end interface layer

  type conectom
    real, allocatable :: m(:,:)
  contains
    procedure :: rand_init => random_connections
  end type conectom

  type network
    type(layer), allocatable :: layers(:)
    type(conectom), allocatable :: weights(:)
    class(Loss), allocatable :: cost_function
  contains
    procedure :: run => interpret
    procedure :: learn => backpropagate
    procedure :: batch_train => batch_train
    procedure :: initiate => random_network
  end type network
  interface network
    module procedure empty_network
  end interface network

  type gradient
    type(conectom), allocatable :: values(:)
  end type gradient

contains

  function layer_derivatives(this) result (out)
    class(layer), intent(in) :: this
    real, dimension(size(this%nodes,1)) :: out
    out = this%func%derivative(this%nodes)
  end function layer_derivatives

  subroutine activate_layer(this, signals)
    class(layer), intent(inout) :: this
    real, dimension(size(this%nodes,1)), intent(in), optional :: signals
    if (present(signals)) then
      this%nodes = signals
    end if
    !keep the last action for bias!
    this%action(1:size(this%nodes,1)) = this%func%activate(this%nodes)
  end subroutine activate_layer

  subroutine passive_activate_layer(this)
    class(layer), intent(inout) :: this
    this%action(1:size(this%nodes,1)) = this%func%activate(this%nodes)
  end subroutine passive_activate_layer

  function empty_network(layout, activations, cost) result (out)
    integer, intent(in) :: layout(:)
    class(wrapper), intent(in) :: activations(:)
    class(Loss), intent(in), optional :: cost
    type(SquaredLoss) :: sqloss
    type(network) :: out
    integer :: i ! a = activatoin length correction

    if (present(cost)) then
      out%cost_function = cost
    else
      out%cost_function = sqloss
    endif

    allocate(out%layers(size(layout, 1)))
    allocate(out%weights(size(layout, 1)-1))
    out%layers(1) = new_layer(layout(1), activations(1)%f)
    do i=2, size(layout,1)
      ! ignore first layer, but access first activation!
      out%layers(i) = new_layer(layout(i), activations(i)%f)
      ! connect PREVIOUS with NEXT layer-size (beware the matrix-mult-rules)
      out%weights(i-1) = new_conectom(layout(i-1),layout(i))
    end do
  end function empty_network

  function new_layer(nodes, activation_function) result (out)
    integer, intent(in) :: nodes
    class(Activation) :: activation_function
    type(layer) :: out

    allocate(out%func, source = activation_function)
    allocate(out%nodes(nodes))
    allocate(out%action(nodes+1))
    out%action(nodes+1) = 1.
  end function new_layer

  subroutine initiate_layer(this)
    type(layer) :: this
    call random_seed()
    call random_number(this%nodes)
    ! DON'T DO RANDOMLY!
    this%action(1:size(this%nodes,1)) = this%func%activate(this%nodes)
  end subroutine initiate_layer

  subroutine random_connections(this)
    class(conectom) :: this
    call random_seed()
    !delivers 0 ... 1 range
    call random_number(this%m)
    !scale to -1 .. 1 range
    this%m = this%m*2 - 1
  end subroutine random_connections

  subroutine random_network(this)
    class(network) :: this
    integer :: i
    do i=1, size(this%weights,1)
      call this%weights(i)%rand_init()
    end do
  end subroutine random_network

  pure subroutine alloc_func(funcout,funcin)
    class(Activation), intent(in) :: funcin
    class(Activation), allocatable, intent(inout) :: funcout
    allocate(funcout, source=funcin)
  end subroutine alloc_func

  pure function layer_from_wrapper(nodes, activation_function) result (out)
    integer, intent(in) :: nodes
    class(wrapper), intent(in) :: activation_function
    type(layer) :: out

    !allocate(out%func, source = activation_function)
    call alloc_func(out%func, activation_function%f)
    allocate(out%nodes(nodes))
    allocate(out%action(nodes+1))
  end function layer_from_wrapper

  pure function new_conectom(incoming, outgoing) result (out)
    integer, intent(in) :: incoming, outgoing
    type(conectom) :: out
    allocate(out%m(outgoing, incoming+1))
  end function new_conectom

  pure function new_conectom_from_layers(incoming, outgoing) result (out)
    type(layer), intent(in) :: incoming, outgoing
    type(conectom) :: out
    allocate(out%m(size(outgoing%nodes,1),size(incoming%action,1)))
  end function new_conectom_from_layers

  function interpret(this, sensor) result (out)
    class(network), intent(inout) :: this
    real, dimension(:),intent(in),optional :: sensor
    real, dimension(size(this%layers(size(this%layers,1))%nodes,1)) :: out
    integer :: i

    if (present(sensor)) then
      call this%layers(1)%activate(sensor)
    else
      call this%layers(1)%activate(this%layers(1)%nodes)
    end if

    do i=1, size(this%weights, 1)
      call this%layers(i+1)%activate(matmul(this%weights(i)%m,this%layers(i)%action))
    end do

    out =      this%layers(size(this%layers))%action

  end function interpret

  pure subroutine signals_to_activation(lyr, signal, activation, derivative)
    type(layer), intent(in) :: lyr
    real, intent(in) :: signal(:,:)
    real, intent(out), dimension(size(signal,1),size(signal,2)) :: activation, derivative
    integer :: i
    do concurrent (i=1:size(signal,2))
      activation(:,i) = lyr%func%activate(signal(:,i))
      derivative(:,i) = lyr%func%derivative(signal(:,i))
    end do
  end subroutine

  subroutine batch_train(this, data, answers, alpha)
    class(network), intent(inout) :: this
    real,intent(in) :: alpha, answers(:,:), data(:,:)
    type(evaluation), dimension(size(this%layers,1)) :: activations, derivatives
    real,allocatable :: tmp(:,:)
    integer :: last,samples
    samples=size(data,2)

    if (.not. allocated(this%cost_function)) stop

    ! assert size(data, 2) == size(answers, 2) !

    last = size(this%layers, 1)
    allocate(tmp, source=data)
    block ! data-collection
      integer :: i,layersize

      do i=1,last-1
        layersize = size(this%layers(i)%nodes,1)

        allocate(activations(i)%vals(layersize+1, samples), source=1.0)
        allocate(derivatives(i)%vals(layersize,samples))
        ! might be faster to set 1 in later step for very large layers?

        call signals_to_activation(            &
          this%layers(i),                      &
          tmp,                                 &
          activations(i)%vals(1:layersize, :), & ! don't overwrite bias 1
          derivatives(i)%vals                  &
        )

        deallocate(tmp)
        allocate(tmp, source=                  &
          matmul(                              &
            this%weights(i)%m,                 &
            activations(i)%vals                &
          )                                    &
        )
        !activations(i)%vals(layersize+1,:) = 1.0/samples
      end do

      !take care of last layer activation
      layersize=size(this%layers(last)%nodes, 1)
      allocate(activations(i)%vals(layersize, samples))
      allocate(derivatives(last)%vals, mold=tmp)
      call signals_to_activation( &
        this%layers(last),        &
        tmp,                      &
        activations(i)%vals,      & ! last layer has no 1 bias
        derivatives(i)%vals       &
      )
      deallocate(tmp)
    end block

    block ! back-propagate
      integer :: i
      real,allocatable :: error(:,:)

      !ERROR = TARGET - PREDICTION
      ! technically 1st derivative of error
      print *, sum(this%cost_function%eval(answers, activations(last)%vals))/size(data,2)
      allocate(error, source=this%cost_function%grad(answers, activations(last)%vals)) !(activations(last)%vals-answers))

      do i=1,last-1
        block
          real :: corrections(size(this%weights(last-i)%m,1),size(this%weights(last-i)%m,2)) ! used weights here to have single-source-of-truth for size
          real, dimension(size(this%layers(last-i)%nodes,1), samples) :: early_error

        ! compute error-contribution of each weight
          ! corr = E x prev_acti.T
          corrections = matmul(        &
            error,                     &
            transpose(                 &
              activations(last-i)%vals & !starts at last-1
            )                          &
          )
          ! corrections(:,layersize+1) == sum(error,2)
          ! since activations include a bottom row of 1s,
          ! this calculates a sum of all errors in the
          ! last collumn. By dividing by samples
          ! (next step) this calculates average error.

        ! compute error-contribution of each node (in previous layer)
          ! Get error of previous layer by back-propagating the current error.
          early_error = matmul(                          & ! initiate matrix multiplication
            transpose(this%weights(last-i)%m             & !  weights.T
              (:, 1:size(this%weights(last-i)%m, 2)-1)), & !      x
            derivatives(last-i+1)%vals * error           & ! error .* g'
          )

          ! overwrite error
          deallocate(error)
          ! "copy to new destination"
          allocate(error, source = early_error)

        !transferre corrections to weights
          this%weights(last-i)%m = &
          this%weights(last-i)%m - &
          (corrections * alpha / samples)

        end block
      end do
    end block
  end subroutine batch_train

  subroutine backpropagate(this, answer, alpha)
    class(network), intent(inout) :: this
    real, intent(in) :: answer(:), alpha
    real, allocatable :: error(:)
    integer :: last, i
    last = size(this%layers, 1)

    ! E = (answer - result)^2 /2
    ! dE/danswer = answer-result

    ! This is the "gradient" of the error-function (i.e. target - result)
    allocate(error, source = answer - this%layers(last)%action(1:size(this%layers(last)%nodes,1)))
    ! or pointer?
    ! error => answer - this%layres ...
    do i=1, last-1

      ! construct new matrix from two error-layer and previous activation:
      ! basically, multiply activation * error pairwise to gauge contribution of weight.
      ! imagine 2 layer, 1 node network, representing y=w*x. Error dependency on w
      ! E = 1/2(T - y)^2, dE/dw = (T-y) * dy/dw = (T-y)*x

      ! in element notation:
      ! w_ij = e_i * a_j  == W = E x A.T

      ! simply adjust biases
      this%weights(last-i)%m(:,size(this%weights(last-i)%m, 2)) = &
      this%weights(last-i)%m(:,size(this%weights(last-i)%m, 2)) + &
      alpha * error

      block ! calculate weight-errors
        real, dimension(1,size(this%layers(last-i)%nodes,1)) :: action_T
        real, dimension(size(this%weights(last-i)%m,1),size(this%weights(last-i)%m,2)-1) :: corrections
        ! this implies a transpose !
        action_T = reshape(this%layers(last-i)%action, [1,size(this%layers(last-i)%nodes,1)])
        !action_T(1,:) = this%layers(last-i-1)%action(1:size(this%layers(last-i-1)%nodes,1))

        ! prepare correction values
        corrections=matmul( &                           ! initate matrix multiplication
          reshape(error, [size(error,1),1]), &          !    error  (turned into n x 1 matrix)
                                                        !      x
          action_T &                                    !   action.T
        )
        !transferre corrections to weights
        this%weights(last-i)%m &
        (:, 1:size(this%weights(last-i)%m,2)-1 ) = & 
        (alpha * corrections)  +  this%weights(last-i)%m &
        (:, 1:size(this%weights(last-i)%m,2)-1 )

      end block

      block ! calculate node-errors
        real, dimension(size(this%layers(last-i)%nodes,1)) :: early_error
        ! Get error of previous layer by back-propagating the current error.
        early_error = matmul( &                          ! initiate matrix multiplication
          transpose(this%weights(last-i)%m &             ! weights.T
            (:, 1:size(this%weights(last-i)%m, 2)-1)), & !     x
          error * this%layers(last-i+1)%derivative() &   ! error .* g'
        )

        ! overwrite error
        deallocate(error)
        ! "copy to new destination" ... i hope ...
        allocate(error, source = early_error)
      end block
    end do

  end subroutine backpropagate

end module network_basics

