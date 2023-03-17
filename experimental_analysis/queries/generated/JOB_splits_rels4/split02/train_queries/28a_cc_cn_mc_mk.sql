SELECT * FROM movie_companies AS mc, company_name AS cn, complete_cast AS cc, movie_keyword AS mk WHERE cn.country_code != '[us]' AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;