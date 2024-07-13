
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
    integer :: window
    class(layer), allocatable :: embedK, embedQ, embedV
    real, allocatable :: embed_K(:,:), embed_Q(:,:), embed_V(:,:), attention(:,:,:), attention_derivatives(:,:,:)
    real, allocatable, dimension(:,:) :: K, Q, VT
    real, allocatable :: QKT(:,:,:), QKT_derivatives(:,:,:)
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

  type, extends(layer) :: dynamic_window
    class(layer), allocatable :: embed_left, embed_right
    logical :: trans_left
    integer :: left_window, right_window
    real, allocatable :: left(:,:,:), right(:,:), interstage(:,:,:)
  contains
    procedure :: train_forward => train_dynamic_window
    procedure :: train_backward => backpropagate_dynamic_window
    procedure :: run => run_dynamic_window
    procedure :: random_init => randomize_dynamic_window
    procedure :: f_init1 => dynamic_window_fixed_init
  end type dynamic_window

  interface dynamic_window
    module procedure dynamic_window_from_layers
    module procedure dynamic_window_from_layers_and_activation
  end interface dynamic_window

contains

  function dynamic_window_from_layers(left_window, left, right_window, right, transpose_left) result (out)
    type(dynamic_window) :: out
    class(layer), intent(in) :: left, right
    integer, intent(in) :: left_window, right_window
    logical, intent(in), optional :: transpose_left
    out%inputs = left%inputs
    allocate(out%embed_left, source=left)
    allocate(out%embed_right, source=right)
    out%in_window = left_window+right_window
    out%left_window = left_window
    out%right_window = right_window
    out%out_window = right_window/right%in_window*right%out_window
    if (present(transpose_left)) then
      out%trans_left = transpose_left
      if (transpose_left) then
        out%outputs = left_window
      else
        out%outputs = left%outputs
      endif
    else
      out%trans_left = .false.
      out%outputs = left%outputs
    endif
  end function dynamic_window_from_layers

  function dynamic_window_from_layers_and_activation(left_window, left, right_window, right, activation_function, transpose_left) result (out)
    type(dynamic_window) :: out
    class(layer), intent(in) :: left, right
    integer, intent(in) :: left_window, right_window
    class(activation), intent(in) :: activation_function
    logical, intent(in), optional :: transpose_left
    out = dynamic_window_from_layers(left_window, left, right_window, right, transpose_left)
    allocate(out%func, source=activation_function)

  end function dynamic_window_from_layers_and_activation

  recursive subroutine randomize_dynamic_window(this)
    class(dynamic_window), intent(inout) :: this
    call this%embed_right%random_init()
    call this%embed_left%random_init()
  end subroutine randomize_dynamic_window

  pure subroutine dynamic_window_fixed_init(this, value)
    class(dynamic_window), intent(inout) :: this
    real, intent(in) :: value
    call this%embed_right%f_init1(value)
    call this%embed_left%f_init1(value)
  end subroutine

  pure function run_dynamic_window(this, signals) result (out)
    class(dynamic_window), intent(in) :: this
    real, intent(in) :: signals(:,:)
    real :: out(this%outputs, size(signals,2)/(this%left_window+this%right_window)*this%right_window)
    integer :: i, samples, window, left_dims(2)

    !preparatory statements

    window=this%left_window+this%right_window
    if (mod(size(signals,2), window) /= 0) error stop "DYNAMIC WINDOW GOT INVALID SIGNAL LENGHT CHUNKS!"
    samples = size(signals,2)/window
    !create new block just to simplify right / left declaration
    left_dims = [this%embed_left%outputs, this%embed_right%outputs * samples]
    if(this%trans_left) then
      left_dims = [this%left_window, this%embed_left%outputs * samples]
    endif

    !The actual function block

    block
      real :: left_signal(this%embed_left%outputs, this%left_window*samples)
      real :: right_signal(this%embed_right%outputs, this%right_window*samples)
      real :: right(this%embed_right%outputs, this%right_window * samples)
      !This definitely works for non-transposed left
      real :: left(left_dims(1), left_dims(2))
      !It works with transposed left as well, assuming SQUARE ATTENTION!
      !If non-square, transposed shape needs to be:
      !real :: left(this%left_window, this%embed_left%outputs * samples) !but this%embed_left%outputs == this%embed_right%outputs

      do concurrent(i=0:samples-1)
        left_signal(:,this%left_window*i+1:this%left_window*(i+1)) = signals(:, window*i+1:window*i+this%left_window)
        right_signal(:,this%right_window*i+1:this%right_window*(i+1)) = signals(:,window*i+this%left_window+1:window*(i+1))
      end do
      right = this%embed_right%run(right_signal)
      if (this%trans_left) then
        left = transpose(this%embed_left%run(left_signal))
      else
        left = this%embed_left%run(left_signal)
      endif
      do concurrent(i=0:samples-1)
        !need to prepare out as concatenated array
        out(:, window*(i-1)+1 : window*i) = matmul(                            &
          left(:,this%embed_right%outputs*i+1:this%embed_right%outputs*(i+1)), &
          right(:,this%right_window*i+1:this%right_window*(i+1))               &
        )

        if (allocated(this%func)) then
          out(:,window*i+1:window*(i+1)) = this%func%activate(out(:,window*i+1:window*(i+1)))
        end if
      end do
    end block

  end function run_dynamic_window

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

  function self_attention_from_params(channels, window, head_size) result (self_attention)
    type(self_attention_head) :: self_attention
    integer, intent(in) :: channels, window, head_size
    self_attention%outputs = head_size
    self_attention%window = window
    self_attention%inputs = channels
    allocate(self_attention%embedK, source = linear_layer(channels, head_size))
    allocate(self_attention%embedQ, source = linear_layer(channels, head_size))
    allocate(self_attention%embedV, source = linear_layer(channels, head_size))
    allocate(self_attention%embed_K(head_size, channels))
    allocate(self_attention%embed_Q(head_size, channels))
    allocate(self_attention%embed_V(head_size, channels))
  end function self_attention_from_params

  function self_attention_from_activation_and_params(func, channels, window, head_size) result (self_attention)
    type(self_attention_head) :: self_attention
    class(Activation), intent(in) :: func
    integer, intent(in) :: channels, window, head_size
    allocate(self_attention%func, source=func)
    self_attention%outputs = head_size
    self_attention%window = window
    self_attention%inputs = channels
    allocate(self_attention%embedK, source = linear_layer(channels, head_size))
    allocate(self_attention%embedQ, source = linear_layer(channels, head_size))
    allocate(self_attention%embedV, source = linear_layer(channels, head_size))
    allocate(self_attention%embed_K(head_size, channels))
    allocate(self_attention%embed_Q(head_size, channels))
    allocate(self_attention%embed_V(head_size, channels))
  end function self_attention_from_activation_and_params

  subroutine self_attention_randomize(this)
    class(self_attention_head), intent(inout) :: this
    call random_number(this%embed_K)
    call random_number(this%embed_Q)
    call random_number(this%embed_V)
    call this%embedK%random_init()
    call this%embedQ%random_init()
    call this%embedV%random_init()
  end subroutine self_attention_randomize

  pure subroutine self_attention_fixed_init(this, value)
    class(self_attention_head), intent(inout) :: this
    real, intent(in) :: value
    this%embed_K=value
    this%embed_Q=value
    this%embed_V=value
    call this%embedK%fixed_init(value)
    call this%embedQ%fixed_init(value)
    call this%embedV%fixed_init(value)
  end subroutine self_attention_fixed_init

  pure function run_self_attention(this, signals) result (out)
    class(self_attention_head), intent(in) :: this
    real, intent(in) :: signals(:,:)
    real, dimension(this%outputs, size(signals,2)) :: out, K, V
    real :: QT(size(signals,2), this%outputs)
    !real, dimension(this%outputs, this%window) :: K, QT, V
    integer :: i,j

    if (modulo(size(signals,2),this%window) /= 0) error stop "SELF ATTENTION GOT INVALID SIGNAL LENGHT CHUNKS!"

    !K = matmul(this%embed_K, signals)
    !QT = transpose(matmul(this%embed_Q, signals))
    !V = matmul(this%embed_V, signals)
    K = this%embedK%run(signals)
    QT = transpose(this%embedQ%run(signals))
    V = this%embedV%run(signals)
    do concurrent (i=0:(size(signals,2)/this%window)-1)
      block
        real :: attention(this%window, this%window)
        attention = matmul(                         &
          K (this%window*i+1:this%window*(i+1), :), &
          QT(:, this%window*i+1:this%window*(i+1))  &
        )

        !exponentiated = exp(attention)
        do concurrent(j=1:this%window)
          block
            real :: exponentiated(j)
            exponentiated = exp(attention(:j,j))
            attention(:j,j)=exponentiated/sum(exponentiated) / (this%window**0.5)
            !attention(:i,i)=exponentiated(:i,i)/sum(exponentiated(:i,i)) / (this%window*1.0)**(0.5)
            attention(j+1:,j)=0
          end block
        end do

        out(:, i+1:i+this%window) = matmul(V(:, i+1:i+this%window), attention)
      end block
    end do

    if (allocated(this%func)) out = this%func%activate(out)

  end function run_self_attention

  recursive function train_dynamic_window(this, signals) result (out)
    class(dynamic_window), intent(inout) :: this
    real, intent(in) :: signals(:,:)
    real :: out(this%outputs, (size(signals,2)/this%in_window)*this%out_window)
    integer :: i, window, samples, left_shape(3)

    !preparatory statements

    window=this%left_window+this%right_window
    if (mod(size(signals,2), window) /= 0) error stop "DYNAMIC WINDOW GOT INVALID SIGNAL LENGHT CHUNKS!"
    samples = size(signals,2)/window

    if(this%trans_left) then
      left_shape = [this%left_window, this%embed_left%outputs, samples]
    else
      left_shape = [this%embed_left%outputs, this%embed_right%outputs, samples]
    endif

    if (allocated(this%interstage)) deallocate(this%interstage)
    allocate(this%interstage(left_shape(1),this%out_window, samples))

    if (allocated(this%func)) then
      if (allocated(this%derivatives)) deallocate(this%derivatives)
      allocate(this%derivatives(this%outputs, (size(signals,2)/this%in_window)*this%out_window))
    endif

    !The actual function block

    !create new block just to simplify right / left declaration
    block
      real :: left_signal(this%embed_left%inputs, this%left_window*samples)
      real :: right_signal(this%embed_right%inputs, this%right_window*samples)
      !This definitely works for non-transposed left
      !It works with transposed left as well, assuming SQUARE ATTENTION!
      !If non-square, transposed shape needs to be:
      !real :: left(this%left_window, this%embed_left%outputs * samples) !but this%embed_left%outputs == this%embed_right%outputs

      !Discecting data to left and right portions!
      do concurrent(i=0:samples-1)
        left_signal(:,this%left_window*i+1:this%left_window*(i+1)) = signals(:, window*i+1:window*i+this%left_window)
        right_signal(:,this%right_window*i+1:this%right_window*(i+1)) = signals(:,window*i+this%left_window+1:window*(i+1))
      end do
      if (allocated(this%right)) deallocate(this%right)
      allocate(this%right, source=this%embed_right%train_forward(right_signal))

      if (allocated(this%left)) deallocate(this%left)
      !Run branch layers (running all at once should be more efficient than running every sample itself)
      !right = this%embed_right%train_forward(right_signal)
      if (this%trans_left) then
        !                            RESHAPE handles slicing and transposing by suitable ORDER !
        allocate(this%left, source = reshape(this%embed_left%train_forward(left_signal), order=[2,1,3], shape=left_shape))
      else
        allocate(this%left, source = reshape(this%embed_left%train_forward(left_signal), shape=left_shape))
      endif

      !Process results
      do concurrent(i=1:samples)
        !Store intermediate result locally
        this%interstage(:,:,i) = matmul(                          &
          this%left(:,:,i),                                       &
          this%right(:,this%out_window*(i-1)+1:this%out_window*i) & !assuming right will produce one output-token for every input-token
        ) ! `right` has it's own out_window different from right_window (which is it's in!)

        !Post process 
        if (allocated(this%func)) then
          !this%derivatives(:, this%right_window*i+1:this%right_window*(i+1)) = &
          !this%func%derivative(          &
          !  this%interstage(:,:,i),      & ! using this%interstage because `out` will mutate. Just safe code practice.
          !  out(:,this%right_window*i+1: & ! passing a SLICE of `out` to derivative to MUTATE!
          !    this%right_window*(i+1))   & ! BEWARE, THIS IS CHANGED HERE!
          !)
          out(:,this%out_window*(i-1)+1 : this%out_window*i) = this%func%activate(this%interstage(:,:,i))
          this%derivatives(:, this%out_window*(i-1)+1 : this%out_window*i) = this%func%derivative(this%interstage(:,:,i))
        else
          out(:, this%out_window*(i-1)+1 : this%out_window*i) = this%interstage(:,:,i)
        end if
      end do
    end block
  end function train_dynamic_window

  recursive function backpropagate_dynamic_window(this, error, alpha) result (out)
    class(dynamic_window), intent(inout) :: this
    real, intent(in) :: error(:,:), alpha
    real :: out(this%inputs, size(error,2)/this%out_window*this%in_window)
    real :: derivative_times_error(size(error,1), size(error,2))
    integer :: i, window, samples, left_width

    !preparatory statements

    window=this%left_window+this%right_window
    if (mod(size(error,2), this%out_window) /= 0) error stop "DYNAMIC WINDOW GOT INVALID SIGNAL LENGHT CHUNKS!"
    samples = size(error,2)/this%out_window

    if(this%trans_left) then
      ! should be relevant for linear layers, so left_window should be correct !
      left_width = this%left_window
    else
      ! transpose not happening for attention matrix, so embed_right%outputs holds # of tokens !
      left_width = this%embed_right%outputs
    endif

    if (allocated(this%func)) then
      derivative_times_error = error * this%derivatives
    else
      derivative_times_error = error
    end if

    ! actual functional block

    block
      real :: left_error(this%embed_left%outputs, left_width/this%embed_left%in_window*this%embed_left%out_window*samples)
      real :: right_error(this%embed_right%outputs, this%right_window/this%embed_right%in_window*this%embed_right%out_window*samples)
      real :: left_error_out(this%embed_left%inputs, left_width*samples)
      real :: right_error_out(this%embed_right%inputs, this%right_window*samples)
      do concurrent(i=1:samples)
        if (this%trans_left) then
          left_error(:,left_width*(i-1)+1:left_width*i) = matmul(                 &
            this%right(:,this%out_window*(i-1)+1:this%out_window*i),              &
            transpose(                                                            &
              derivative_times_error(:,this%out_window*(i-1)+1:this%out_window*i) &
            )                                                                     &
          )
        else
          left_error(:,left_width*(i-1)+1:left_width*i) = matmul(                &
            derivative_times_error(:,this%out_window*(i-1)+1:this%out_window*i), &
            transpose(                                                           &
              this%right(:,this%out_window*(i-1)+1:this%out_window*i)            &
            )                                                                    &
          )
        endif
        right_error(:,this%out_window*(i-1)+1:this%out_window*i) = matmul( &
          transpose(this%left(:,:,i)),                               &
          derivative_times_error(:,             &
            this%out_window*(i-1)+1:this%out_window*i &
          )                                              &
        )
      end do

      left_error_out = this%embed_left%train_backward(left_error, alpha)
      right_error_out = this%embed_right%train_backward(right_error, alpha)

      do concurrent(i=0:samples-1)
        out(:, i*window+1 : i*window+this%left_window) = left_error_out(:,i*this%left_window+1:(i+1)*this%left_window)
        out(:, i*window+1+this%left_window:(i+1)*window) = right_error_out(:,i*this%right_window+1:(i+1)*this%right_window)
      end do
    end block
  end function backpropagate_dynamic_window

  function train_forward_self_attention(this, signals) result (out)
    class(self_attention_head), intent(inout) :: this
    real, intent(in) :: signals(:,:)
    real, dimension(this%outputs, size(signals,2)) :: out, K, V
    real :: QT(size(signals,2),this%outputs)
    real :: inv_sqrt
    integer :: i,j, samples

    if (mod(size(signals,2),this%window) /= 0) error stop "SELF ATTENTION GOT INVALID SIGNAL LENGHT CHUNKS!"

    samples=size(signals,2)/this%window ! we know this works because of IF above

    K = matmul(this%embed_K, signals)
    V = matmul(this%embed_V, signals)
    allocate(this%K, source=K)
    allocate(this%VT, source=transpose(V))
    allocate(this%Q, source=matmul(this%embed_Q, signals))
    QT = transpose(this%Q)

    inv_sqrt = this%window**0.5/this%window ! for numerical stability'

    if (allocated(this%signals)) deallocate(this%signals)
    allocate(this%signals, source=signals)
    if (allocated(this%attention)) deallocate(this%attention)
    allocate(this%attention(this%window,this%window,samples))
    if (allocated(this%attention_derivatives)) deallocate(this%attention_derivatives)
    allocate(this%attention_derivatives(this%window,this%window,samples),source=0.)

    do concurrent(i=0:samples-1)
      this%attention(:,:,i+1) = matmul( &
          QT(this%window*i+1:this%window*(i+1), :), &
          K (:, this%window*i+1:this%window*(i+1))  &
      )
      do concurrent(j=1:this%window)
        block
          real :: exponentiated(j), sum_exp
          exponentiated = exp(this%attention(:j,j,i+1))
          sum_exp = sum(exponentiated)
          this%attention(:j,j,i+1)=exponentiated/sum_exp * inv_sqrt
          this%attention(j+1:,j,i+1)=0

          this%attention_derivatives(:j,j,i+1) = (sum_exp-exponentiated)*exponentiated/sum_exp * inv_sqrt
        end block
      end do

      out(:, 1+i*this%window:(1+i)*this%window) = matmul(V(:,1+i*this%window:(1+i)*this%window),this%attention(:,:,i+1))
    end do

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
    real :: errorQ(size(this%embed_Q,1),size(error,2))
    real :: errorK(this%outputs,size(error,2))
    real :: errorV(this%outputs,size(error,2))
    real :: out(this%inputs,size(error,2)), derivative_times_error(this%outputs, size(error,2))
    integer :: i, samples, channels(size(error,2))
    real :: QKT_err(this%outputs, size(this%embed_V,1))

    if (mod(size(error,2),this%window) /= 0) error stop "SELF ATTENTION GOT INVALID SIGNAL LENGHT CHUNKS!"

    samples = size(error,2) / this%window ! works because of above IF statement

    if (allocated(this%func)) then
      derivative_times_error = error*this%derivatives
    else
      derivative_times_error = error
    end if

    do concurrent (i=0:samples-1)
      block
        real :: attention_error(this%window, this%window)
        attention_error = matmul(                                     &
          this%VT(this%window*i+1:this%window*(i+1), :),              &
          derivative_times_error(:,this%window*i+1:this%window*(i+1)) &
        )
        errorV(:,this%window*i+1:this%window*(i+1)) = matmul(          &
          derivative_times_error(:,this%window*i+1:this%window*(i+1)), &
          transpose(this%attention(:,:,i+1))                           &
        )
        errorQ(:,this%window*i+1:this%window*(i+1)) = matmul( &
          this%K(:,this%window*i+1:this%window*(i+1)),        &
          transpose(attention_error)                          &
        )
        errorK(:,this%window*i+1:this%window*(i+1)) = matmul( &
          this%Q(:,this%window*i+1:this%window*(i+1)),        & ! transpose of QT is Q
          attention_error                                     &
        )
      end block
    end do

    out = errorV+errorQ+errorK

    block ! calculate actual correction matrices
      real :: corrections_K(size(this%embed_K,1),size(this%embed_K,2))
      real :: corrections_Q(size(this%embed_Q,1),size(this%embed_Q,2))
      real :: corrections_V(size(this%embed_V,1),size(this%embed_V,2))
      ! need do convert errors into corrections, a bunch of matrix multiplications ...
      corrections_K = matmul(   &
        errorK,                 &
        transpose(this%signals) &
      ) / size(error,2)
      corrections_Q = matmul(   &
        errorQ,                 &
        transpose(this%signals) &
      ) / size(error,2)
      corrections_V = matmul(   &
        errorV,                 &
        transpose(this%signals) &
      ) / size(error,2)
      this%embed_Q = this%embed_Q - corrections_Q*alpha
      this%embed_K = this%embed_K - corrections_K*alpha
      this%embed_V = this%embed_V - corrections_V*alpha

    end block
    deallocate(this%K, this%VT, this%Q)

  end function backpropagate_self_attention

end module combined_layers