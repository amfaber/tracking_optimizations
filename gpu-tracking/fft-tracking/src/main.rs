use std::sync::mpsc;
use std::thread;


trait MyTrait{
    fn my_func(&self) -> i32;
}

impl MyTrait for u32{
    fn my_func(&self) -> i32{
        5
    }
}

fn func<T: MyTrait + Send>(x: T){
    let (tx, rx) = mpsc::channel::<T>();
    tx.send(x).unwrap();
    thread::scope(|s| {
        let handle = s.spawn(move || {
            let idk = rx.recv().unwrap();
            idk.my_func()
        });
        handle.join().unwrap()
    });
}

fn main() {
    let x = 1u32;
    func(x);
}
