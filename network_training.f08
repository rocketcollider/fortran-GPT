
program network_training
  use structured_network
  use network_layers
  use file_helpers
  implicit none
  !use network_basics

  block ! binary-test
    type(network) :: net
    real :: answers(8,256),data(256,256)
    real ::  cost
    integer :: i(8), index, run, answer

    type(linear_layer) :: simple_lyr
    type(SquaredLoss):: sqloss
    ! type(ReLU), target :: ReLUf
    ! type(softmax), target :: soft
    data = 0.
    answers = 0.

    associate (i1=>i(1), i2=>i(2), i3=>i(3), i4=>i(4), i5=>i(5), i6=>i(6), i7=>i(7), i8=>i(8))
      do i1=0,1
        do i2=0,1
          do i3=0,1
            do i4=0,1
              do i5=0,1
                do i6=0,1
                  do i7=0,1
                    do i8=0,1
                      index = sum(i * 2**[7,6,5,4,3,2,1,0])+1
                      data(index,index)=1.
                      answers(:,index) = i
                    end do
                  end do
                end do
              end do
            end do
          end do
        end do
      end do
    end associate

    do run=1,10
      simple_lyr = linear_layer(256,8)
      net = network([simple_lyr], sqloss)
      !net = network([256,8],[wrap(ident), wrap(ident)])
      call net%layers(1)%fixed_init( 0.1 )
      answer = 1000
      do index=1,100
        cost = net%batch_train(data, answers, 1.9825)
      end do

      cost = sum(sqLoss%eval(net%run(data), answers))/size(answers,2)

      if (cost > 0.5) then
        print *, 'binomial training failed at'
        print *, answer
        print *, "step, during run: "
        print *, run
        print *, "actual output: "
        print *, net%run(data(:,27))
      end if
    end do
  end block

  block ! simple perceptron
    real, dimension(1,1001) :: data, answers
    type(network) :: net
    type(identity) :: ident
    type(ReLU) :: ReLUf
    type(SquaredLoss) :: sqLoss
    integer :: i
    real :: cost

    do i=1,1001
      data(1,i)=-2.+4*(i-1)/1000.
      answers(1,i)= data(1,i)**2
    end do
    print *, data(1,1), answers(1,1)
    net = network([1,5,1], [bias_layer(ReLUf), bias_layer(ident)], sqLoss)
    !call net%initiate()
    call net%layers(1)%fixed_init(0.1)
    call net%layers(2)%fixed_init(0.5)
    print *, sum(data)/size(data)
    do i=1,10
      cost = net%batch_train(data, answers, 0.6)
      print *, cost
    end do
    print *, net%run([-2.]),net%run([-1.]),net%run([0.]),net%run([1.]),net%run([2.])

  end block

  block ! ingest training data
    integer :: io, strlen
    character(len=128) :: a
    10 format(A)
    print *, ' plain file open'
    io = 100
    open(newunit=io, file='names.txt', status='old', action='read')
    print *, io
    read(io, 10) a
    do strlen=128,1,-1
      if (a(strlen:strlen) /= ' ') then
        exit
      end if
    end do
    close(io)
    print *, a
  end block

  block
    type(file) names
    character(len=:), allocatable :: line
    integer, allocatable :: out(:)
    type(network) :: net
    integer :: dic_len

    print *, "before declaraton"
    names = open_path('names.txt')
    print *, "after declaration"
    !allocate(line, source=names%readline())
    line = names%readline()
    line = names%readline()

    allocate(out, source=names%dictionarize())

    dic_len=len(names%dictionary)

    block ! train network
      !type(identity), target :: ident
      type(softmax),target :: sftmax
      type(LogLoss) :: cost
      type(bias_layer) :: simple_match
      real :: training_set(dic_len+1, size(out)), rest(dic_len+1, dic_len+1), loss_vals
      integer :: i,test(dic_len+1, dic_len+1), last, next

      test = 0

      do i=1,size(out)-1
        test(out(i)+1, out(i+1)+1) = (test(out(i)+1, out(i +1) +1) +1) !brackets just for alliteration
      end do
      print *, names%dictionary
      print *, test(1,:)
      print *, test(4,:)

      do i=1, dic_len+1
        rest(i,:) = test(i,:)*1.0/sum(test(i,:))
        !print *, rest(i,:)
      end do

      call srand(12345)
      do i=1,10 !generate 10 names
        last = 1
        line = ''
        do while (.true.)
          next = random_select(rest(last,:))
          if (next == 1) then
            if (len(line)>3) then
              exit
            else
              cycle
            end if
          end if
          line = line // names%dictionary(next-1:next-1)
          print *, line
          last = next
        end do
      end do

      training_set=0

      simple_match = bias_layer(sftmax, dic_len+1, dic_len+1)
      net = network([simple_match], cost)
      !net = network([dic_len+1,dic_len+1], [wrap(ident), wrap(sftmax)], cost)
      call net%initiate()

      do concurrent (i=1:size(out))
        training_set(:,i)=one_hot(out(i)+1,dic_len+1)
      end do

      do i=1,1
        loss_vals = net%batch_train( training_set(: , :size(out)-1), training_set(:,2:), 0.2)
      end do
      print *, 'end'

    end block
    block
      real :: data(2,3,1), out(2,3,1)
      data = reshape([1,1,2,3,3,2],[2,3,1])
      print *, data
      out = rolling_average(data)
      print *, out
    end block

  end block

end program network_training