---
title: "Signal's Safety Number" 
---

# [Signal's Safety Number](https://Signal.org/blog/safety-number-updates/)

## Basics

Let's put the safety number in terms of PGP because really there have
been no innovations in the basic cryptographic principles since PGP.

Let's say my fingerprint is 0xA99DD60E and your fingerprint is 0x035E72CD.
A "fingerprint" is a hash of the public key --- it (in theory) uniquely
identifies the public key (note that brute force attacks ARE feasible
for PGP, so an attacker can generate keys until the fingerprint happens
to match, so if you `gpg --verify` it looks like I signed something
but in fact it's the attacker's key whose fingerprint matches mine ...
this is currently preventable by giving the full fingerprint, i.e.
0xEA6E27948C7DBF5D0DF085A10FBC2E3BA99DD60E)

So when I send a message to you, I encrypt it with the public key associated
with the fingerprint 0x035E72CD, but suppose I don't know that 0x035E72CD
is your actual fingerprint, instead I've been encrypting all my messages
to 0x42EFA11D. 0x42EFA11D belongs to the NSA, who has been snooping on the
connection (man-in-the-middle). When I send a message to "you", it in fact goes
through a NSA server, who owns the public key 0x42EFA11D. The NSA then decrypts
the message, writes it down, and then re-encrypts the message with your public
key. You receive the message, encrypted, nothing seems wrong to either of us.

To prevent this attack, I need to make sure that what I think is your
public key really is your public key, and symmetrically, you need to make
sure that what you think is my public key really is my public key. For
PGP, either you meet up in person, talk through some secure channel, or
believe in the web of trust (which verifies keys if you trust people who
trust my key, graph algorithm). The thing about Signal is that its users
are not well-versed in public key/private key cryptography, so they cannot
be trusted to maintain keys. An old version of Signal showed your public
key and what you thought their public key was, so you'd have 2 pairs of
fingerprints, or 4 distinct things and people were really confused what to
check against what. They intelligently simplified this paradigm by simply
concatenating the two fingerprints, so checking the one number I have against
the one number you have simultaneously verifies that what I think is your
key is really your key and what you think my key is really is my key (note
that this per-conversation Signal number doesn't even add additional work
compared to showing a per-user key, since for each conversation you need
to check their key anyways). Note that since a Signal safety number is
simply the two people's fingerprints concatenated, you can find your own
public key by just checking two different safety numbers and finding the
common denominator (which block of 6 words of 5 numbers is the same).

## Verification

Now this brings us to possible verification mechanisms.

It'd be really stupid for both of us to just post the safety number
in Signal and then compare, since if we assume the NSA is snooping
they'd just manipulate what is being shown on either end so both of
us are deceived. Therefore you need to do something like `gpg --sign`
to make it impossible to modify the message without detection.

First off, it's perfectly safe to share safety numbers because they're
simply our two fingerprints concatenated together. A fingerprint derives
from the public key, which is meant to be shared. Second, if you post the
safety number in public then they can figure out you're talking to me,
since your key is there and my key is there. `gpg --sign`'ing a safety
number therefore _proves_ that you're in communication with me (i.e. not
a good idea to `gpg --sign` a safety number with a terrorist since it'll
be really hard to prove that you didn't).

Assuming your PGP key hasn't been compromised, then you signing
the safety number means that it really is your fingerprint and
what you think my fingerprint is. Likewise, if I sign and then
the safety number matches, we're all good.

## Summary

- If the safety numbers match, that prevents a man-in-the-middle attack and
guarantees who you think you're talking to really is who you're talking to
    - Note that if they don't match, it could be a result of uninstalling
    Signal, changing phones, etc. --- Signal does not associate
    a permanent key per account, it changes on some operations.
- Sharing a Signal safety number in public is safe
(since it shares fingerprints) but is showing to
the world that you're talking to the other person
    - Similarly, if you `gpg --sign` a safety number, you
    prove you're in correspondence with that person
- Like PGP, you need a trusted channel to exchange safety numbers
    - i.e. you have confidence that you're talking to the right person
    - This channel cannot be Signal itself, for obvious reasons
    - Comparing `gpg --sign`'ed safety numbers is a good guarantee
- If you want, you can get your fingerprint by comparing the safety
numbers of two different conversations. Once you know your fingerprint
you know the fingerprint of each of the people you're talking to.

